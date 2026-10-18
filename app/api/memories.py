"""Memory browsing and explicitly reviewed manual corrections."""
import json
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from app.agent import outcome_memory
from app.memory import memory_corrections
from app.store import repo
from app.store.db import get_conn

router = APIRouter(prefix='/api')

MemoryKind = Literal['semantic', 'episodic', 'reflective', 'journal', 'skill']
ProposalStatus = Literal['all', 'pending', 'confirmed', 'discarded']

_VISIBLE_MEMORY = "m.deleted_at IS NULL AND (m.person_id IS NULL OR " \
                  "(p.id IS NOT NULL AND p.deleted_at IS NULL AND p.status!='merged'))"
_MEMORY_FROM = ' FROM memory_item m LEFT JOIN person p ON p.id=m.person_id '
_EVIDENCE_COLUMNS = ','.join(f'e.{key} AS source_{key}' for key in
                            ('id', 'kind', 'source_ref', 'excerpt', 'blob_hash', 'created_at'))
_SOURCE_KEYS = ('id', 'kind', 'source_ref', 'excerpt', 'blob_hash', 'created_at')
_ORIGIN_COLUMNS = ','.join(f'o.{key} AS origin_{key}' for key in _SOURCE_KEYS)
_MEMORY_SELECT = 'SELECT m.*,p.display_name AS person_name,' + _EVIDENCE_COLUMNS + ',' + \
    _ORIGIN_COLUMNS + ',(SELECT COUNT(*) FROM memory_correction c WHERE c.memory_id=m.id) AS correction_count'
_MEMORY_JOINS = _MEMORY_FROM + 'LEFT JOIN evidence e ON e.id=m.evidence_id ' + \
    'LEFT JOIN evidence o ON o.id=m.origin_evidence_id'


class CorrectIn(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    action: Literal['edit', 'withdraw', 'restore']
    expected_revision: StrictInt = Field(ge=0)
    request_id: StrictStr = Field(min_length=1, max_length=128)
    reason: StrictStr = Field(default='', max_length=1000)
    text: StrictStr | None = Field(default=None, min_length=1, max_length=20000)
    valid_from: StrictStr | None = None
    person_id: StrictStr | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode='after')
    def edits(self):
        changed = self.model_fields_set & {'text', 'valid_from', 'person_id'}
        if self.action != 'edit' and changed:
            raise ValueError('撤销和恢复不能附带内容、日期或归属修改')
        if self.action == 'edit' and not changed:
            raise ValueError('请至少提交内容、日期或归属之一')
        if not self.request_id.strip() or ('text' in changed and (self.text is None or not self.text.strip())) \
                or ('valid_from' in changed and self.valid_from is None) \
                or ('person_id' in changed and self.person_id is not None and not self.person_id.strip()):
            raise ValueError('请提交有效的请求标识、内容、日期或人物归属')
        return self


def _person_filter(conn, person_id):
    if person_id is None:
        return
    person = repo.get_person(conn, person_id)
    if not person or person['deleted_at'] is not None or person['status'] == 'merged':
        raise HTTPException(404, '人物不存在或已归档')


def _person_search(person_expression):
    return f"instr(lower(COALESCE(p.display_name,'')),lower(?))>0 OR EXISTS " \
           f"(SELECT 1 FROM person_alias a WHERE a.person_id={person_expression} " \
           "AND instr(lower(a.alias),lower(?))>0)"


def _source(row, prefix='source_', truncate=False):
    if row[prefix + 'id'] is None:
        return None
    source = {key: row[prefix + key] for key in _SOURCE_KEYS}
    if truncate and isinstance(source['excerpt'], str):
        source['excerpt'] = source['excerpt'][:300]
    return source


def _confirmed_links(conn, memory_ids):
    if not memory_ids:
        return {}
    slots = ','.join('?' for _ in memory_ids)
    original_columns = ','.join(f'e.{key} AS original_{key}' for key in _SOURCE_KEYS)
    rows = conn.execute(
        "SELECT i.id AS proposal_id,json_extract(i.payload,'$.followup_id') AS followup_id,"
        "json_extract(i.payload,'$.receipt.id') AS memory_id," + original_columns +
        ' FROM pending_item i LEFT JOIN outcome_memory_analysis a '
        "ON a.outcome_id=json_extract(i.payload,'$.outcome_id') "
        'LEFT JOIN evidence e ON e.id=a.evidence_id '
        "WHERE i.kind='outcome_memory' AND i.resolution='confirmed' AND i.resolved_at IS NOT NULL "
        "AND json_extract(i.payload,'$.status')='confirmed' "
        "AND json_extract(i.payload,'$.kind')='memory' "
        "AND json_extract(i.payload,'$.receipt.kind')='memory' "
        f"AND json_extract(i.payload,'$.receipt.id') IN ({slots}) "
        'ORDER BY i.resolved_at DESC,i.created_at DESC,i.id DESC', memory_ids).fetchall()
    links = {}
    for row in rows:
        links.setdefault(row['memory_id'], row)
    return links


def _memory(row, link=None, truncate=False, conn=None):
    source = _source(row, truncate=truncate)
    origin_source = _source(row, prefix='origin_', truncate=truncate)
    original_source = None
    if link and source and (source['source_ref'] == 'outcome-memory-review:' + link['proposal_id']
                           or (source['source_ref'] or '').startswith('memory-correction:')):
        original_source = _source(link, prefix='original_', truncate=truncate)
    if original_source is None and origin_source and conn is not None \
            and (origin_source['source_ref'] or '').startswith('outcome-memory-review:'):
        try:
            original_id = json.loads(row['origin_excerpt']).get('source_evidence_id')
        except (json.JSONDecodeError, TypeError, AttributeError):
            original_id = None
        if isinstance(original_id, str):
            evidence = conn.execute('SELECT * FROM evidence WHERE id=?', (original_id,)).fetchone()
            if evidence:
                original_source = {key: evidence[key] for key in _SOURCE_KEYS}
                if truncate and isinstance(original_source['excerpt'], str):
                    original_source['excerpt'] = original_source['excerpt'][:300]
    return {key: row[key] for key in ('id', 'kind', 'text', 'ts', 'created_at', 'confidence', 'status',
        'revision', 'assignment_revision', 'updated_at', 'correction_count')} | {
        'person': {'id': row['person_id'], 'display_name': row['person_name']}
            if row['person_id'] is not None else None,
        'source': source, 'origin_source': origin_source, 'original_source': original_source,
        'proposal_id': link['proposal_id'] if link else None,
        'followup_id': link['followup_id'] if link else None}


@router.get('/memories')
def memories(q: str = Query(default='', max_length=200), person_id: str | None = None,
             scope: Literal['all', 'personal'] = 'all', kind: MemoryKind | None = None,
             status: Literal['active', 'withdrawn', 'all'] = 'active',
             limit: int = Query(default=30, ge=1, le=100), offset: int = Query(default=0, ge=0)):
    if scope == 'personal' and person_id is not None:
        raise HTTPException(422, '个人记忆范围与人物筛选不能同时使用')
    conn = get_conn()
    _person_filter(conn, person_id)
    where, params = [_VISIBLE_MEMORY], []
    if status != 'all':
        where.append('m.status=?')
        params.append(status)
    if scope == 'personal':
        where.append('m.person_id IS NULL')
    elif person_id is not None:
        where.append('m.person_id=?')
        params.append(person_id)
    if kind is not None:
        where.append('m.kind=?')
        params.append(kind)
    if q:
        where.append('(instr(lower(m.text),lower(?))>0 OR ' + _person_search('m.person_id') + ')')
        params.extend((q, q, q))
    clause = ' WHERE ' + ' AND '.join(where)
    total = conn.execute('SELECT COUNT(*)' + _MEMORY_FROM + clause, params).fetchone()[0]
    rows = conn.execute(_MEMORY_SELECT + _MEMORY_JOINS + clause +
        ' ORDER BY m.ts DESC,m.created_at DESC,m.id DESC LIMIT ? OFFSET ?',
        (*params, limit, offset)).fetchall()
    links = _confirmed_links(conn, [row['id'] for row in rows])
    return {'memories': [_memory(row, links.get(row['id']), truncate=True, conn=conn) for row in rows],
            'total': total, 'has_more': offset + len(rows) < total, 'limit': limit, 'offset': offset}


@router.get('/memories/{memory_id}')
def memory_detail(memory_id: str):
    conn = get_conn()
    row = conn.execute(_MEMORY_SELECT + _MEMORY_JOINS + ' WHERE ' + _VISIBLE_MEMORY +
        ' AND m.id=?', (memory_id,)).fetchone()
    if not row:
        raise HTTPException(404, '记忆不存在或已不可见')
    return _memory(row, _confirmed_links(conn, [memory_id]).get(memory_id), conn=conn)


@router.post('/memories/{memory_id}/correct')
def correct(memory_id: str, body: CorrectIn):
    conn = get_conn()
    try:
        result = memory_corrections.correct_memory(conn, memory_id, body.model_dump(exclude_unset=True))
    except memory_corrections.MemoryCorrectionError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    result['memory'] = memory_detail(memory_id)
    return result


@router.get('/memories/{memory_id}/history')
def history(memory_id: str):
    try:
        return {'history': memory_corrections.history(get_conn(), memory_id)}
    except memory_corrections.MemoryCorrectionError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.get('/memory-updates')
def memory_updates(q: str = Query(default='', max_length=200), person_id: str | None = None,
                   status: ProposalStatus = 'all', kind: Literal['field', 'memory'] | None = None,
                   limit: int = Query(default=30, ge=1, le=100), offset: int = Query(default=0, ge=0)):
    conn = get_conn()
    _person_filter(conn, person_id)
    person_expression = "json_extract(i.payload,'$.person_id')"
    from_clause = ' FROM pending_item i JOIN person p ON p.id=' + person_expression + \
        ' JOIN outcome_memory_analysis a ON a.outcome_id=json_extract(i.payload,\'$.outcome_id\')'
    where = ["i.kind='outcome_memory'", "p.deleted_at IS NULL", "p.status!='merged'"]
    params = []
    if person_id is not None:
        where.append(person_expression + '=?')
        params.append(person_id)
    if status != 'all':
        where.append("json_extract(i.payload,'$.status')=?")
        params.append(status)
    if kind is not None:
        where.append("json_extract(i.payload,'$.kind')=?")
        params.append(kind)
    if q:
        where.append("(instr(lower(json_extract(i.payload,'$.content')),lower(?))>0 OR "
                     "instr(lower(json_extract(i.payload,'$.evidence')),lower(?))>0 OR " +
                     _person_search(person_expression) + ')')
        params.extend((q, q, q, q))
    clause = ' WHERE ' + ' AND '.join(where)
    total = conn.execute('SELECT COUNT(*)' + from_clause + clause, params).fetchone()[0]
    rows = conn.execute('SELECT i.id' + from_clause + clause +
        ' ORDER BY i.created_at DESC,i.id DESC LIMIT ? OFFSET ?', (*params, limit, offset)).fetchall()
    proposals = []
    for row in rows:
        try:
            proposals.append(outcome_memory.read_proposal(conn, row['id']))
        except outcome_memory.OutcomeMemoryError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
    return {'proposals': proposals, 'total': total, 'has_more': offset + len(proposals) < total,
            'limit': limit, 'offset': offset}
