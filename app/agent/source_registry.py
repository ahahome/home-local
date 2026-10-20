"""Turn-local references are assigned only to records read from storage."""
from collections import Counter
from copy import deepcopy
from html.parser import HTMLParser
import json
import re

from markdown_it import MarkdownIt

from app.agent.followup import _atomic
from app.agent.memory_sources import source_state

_REFERENCE = re.compile(r'\[\[([A-Za-z0-9_][A-Za-z0-9_.:-]*)\]\]')
_MARKDOWN = MarkdownIt('gfm-like')


class _CitationText(HTMLParser):
    excluded = {'a', 'code', 'pre', 'script', 'style', 'iframe', 'textarea', 'title', 'template'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocked = []
        self.references = []

    def handle_starttag(self, tag, attrs):
        if tag in self.excluded:
            self.blocked.append(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.blocked) - 1, -1, -1):
            if self.blocked[index] == tag:
                del self.blocked[index:]
                break

    def handle_data(self, data):
        if not self.blocked:
            self.references.extend(_REFERENCE.findall(data))


def _prose_reference_occurrences(answer):
    parser = _CitationText()
    parser.feed(_MARKDOWN.render(answer))
    parser.close()
    return parser.references


def _prose_references(answer):
    return list(dict.fromkeys(_prose_reference_occurrences(answer)))


class SourceChangedError(ValueError):
    pass


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def compact_source(source):
    from app.agent.memory_sources import model_evidence_context, model_source_context

    value = {key: source[key] for key in (
        'id', 'type', 'entity_id', 'person', 'text', 'people') if key in source}
    value.update(model_source_context(source))
    value['text'] = value.get('text', '')[:1600]
    if source.get('type') in {'memory', 'field', 'interaction'}:
        # Storage classifications do not establish whether text is verbatim.
        # Known audit envelopes also contain dates named after API input keys;
        # their full forms remain in the quotation but never enter model input.
        value['文字依据'] = model_evidence_context(source)
    if source.get('type') == 'field':
        field = source.get('field')
        labels = {'company': '公司', 'title': '职位', 'city': '城市', 'birthday': '生日',
                  'phone': '电话', 'email': '电子邮箱', 'family': '家庭'}
        if isinstance(field, str):
            value['档案项目'] = labels.get(field, field)
            if field.startswith(('trait.', 'need.')):
                prefix, name = field.split('.', 1)
                value['档案项目'] = ('特点：' if prefix == 'trait' else '需求：') + name
    return value


class SourceRegistry:
    def __init__(self, conn):
        self.conn = conn
        self.sources = {}
        self._keys = {}
        self._profiles = {}

    @property
    def dependencies(self):
        return deepcopy([*self.sources.values(), *self._profiles.values()])

    def register_profile(self, person_id, card):
        snapshot = {'type': 'profile_context', 'entity_id': person_id, 'text': card}
        self._profiles[_json(snapshot)] = snapshot

    def _key(self, source):
        return _json({key: value for key, value in source.items() if key not in {'id', 'captured_at'}})

    def preview(self, source):
        sid = self._keys.get(self._key(source), f'S{len(self.sources) + 1}')
        return {**deepcopy(source), 'id': sid}

    def preview_many(self, sources):
        """Assign sequential IDs for a whole group without registering any source.

        Budget checks must inspect the same compact group that a successful
        registration would produce. Duplicates and already registered records
        reuse their IDs, including duplicates first encountered in this group.
        """
        keys = dict(self._keys)
        next_id = len(self.sources) + 1
        values = []
        for source in sources:
            key = self._key(source)
            sid = keys.get(key)
            if sid is None:
                sid = f'S{next_id}'
                keys[key] = sid
                next_id += 1
            values.append({**deepcopy(source), 'id': sid})
        return values

    def register(self, source):
        value = self.preview(source)
        if value['id'] in self.sources:
            # The key ignores captured_at. Re-reading unchanged content must not
            # replace the original turn-local quotation with a later capture.
            return deepcopy(self.sources[value['id']])
        self._keys[self._key(source)] = value['id']
        self.sources[value['id']] = value
        return value

    def tool_result(self, result, budget=12000):
        if 'card' not in result and not result.get('sources') and not any(
                isinstance(item, dict) and item.get('source') for item in result.get('results', [])):
            return result
        output = {key: deepcopy(value) for key, value in result.items() if key not in {'results', 'sources'}}
        if len(_json({**output, 'sources': [], 'results': [], 'truncated': True})) > budget:
            return {'ok': False, 'error': '本轮资料已达到上下文预算，请缩小范围或新建对话后重试。',
                    'truncated': True}
        if result.get('card') and result.get('person_id'):
            self.register_profile(result['person_id'], result['card'])
        for key in ('sources', 'results'):
            if key not in result:
                continue
            output[key] = []
            for item in result[key]:
                source = item if key == 'sources' else item.get('source')
                candidate = compact_source(self.preview(source)) if source else None
                entry = candidate if key == 'sources' else {**item, 'text': item.get('text', '')[:1600]}
                if key == 'results' and source:
                    # The model gets the labelled dates in the source projection.
                    # Raw duplicate timestamps otherwise look like edit times.
                    entry.pop('ts', None)
                    entry.pop('when', None)
                    entry.pop('kind', None)
                    entry.update(source=candidate, source_id=candidate['id'])
                if len(_json({**output, key: output[key] + [entry], 'truncated': True})) > budget:
                    output['truncated'] = True
                    break
                if source:
                    self.register(source)
                output[key].append(entry)
        return output

    def citations(self, answer, require_citations=False):
        occurrences = _prose_reference_occurrences(answer)
        referenced = list(dict.fromkeys(occurrences))
        used = [sid for sid in referenced if sid in self.sources]
        invalid = set(referenced) - self.sources.keys()
        prose_counts = Counter(occurrences)
        raw_counts = Counter(_REFERENCE.findall(answer))
        # Remove an identifier only when every occurrence is visible prose.
        # Escapes/entities can synthesize markers and make these counts ambiguous.
        removable = {sid for sid in invalid if raw_counts[sid] == prose_counts[sid]} if not (
            '\\' in answer or '&' in answer) else set()
        answer = _REFERENCE.sub(lambda match: '' if match.group(1) in removable else match.group(0), answer)
        warnings = []
        if invalid:
            warnings.append({'code': 'invalid_citation', 'message': '部分资料引用无法核验，未列为资料来源，请核对回答。'})
        if not used and require_citations and self.sources:
            warnings.append({'code': 'uncited', 'message': '本条回答未附可核查的资料引用，请核对其中的人物信息。'})
        if any(self.sources[sid]['temporal_status'] in {'historical_background', 'historical_field'} for sid in used):
            warnings.append({'code': 'historical_context', 'message': '引用包含有日期的历史背景，当前是否仍适用需要核实。'})
        return answer, [deepcopy(self.sources[sid]) for sid in used], warnings

    def save_answer(self, save):
        # Hold the write lock across final source checks and the answer insert.
        with _atomic(self.conn):
            if any(source_state(self.conn, source)['status'] != 'current' for source in self.dependencies):
                raise SourceChangedError('引用资料在生成期间发生变化，请重试以读取最新记录')
            return save()
