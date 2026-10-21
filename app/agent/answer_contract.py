"""Choose presentation permissions, independently of permission to write memory.

This conservative language heuristic is a UX decision, not a semantic verifier.
Unrecognized requests remain record queries; they never gain write permission.
"""
import re

_GENERATION = re.compile(
    r'起草|草拟|拟[一几条份段个封]|拟写|代[写笔]|写(?:一|个|条|段|封|份|给)|'
    r'回复(?:一|他|她|对方)|开场白|话术|文案|措辞|'
    r'给(?:我|出|点|些|一|个).{0,12}(?:建议|方案|步骤)|建议.{0,8}(?:怎么|如何)|'
    r'(?:怎么|如何).{0,12}(?:建议|准备|开口|沟通|跟进|处理)|'
    r'要不要|怎么办|帮我.{0,12}(?:分析|规划|安排|准备)|'
    r'聊聊|倾诉|安慰我|陪我聊|'
    r'\b(?:draft|compose|rewrite|suggest|advice|recommend|how\s+(?:should|can|to))\b', re.I)
# Text operations are requests even when the object is the previous answer.
# Anchor imperative forms so questions about an already edited record stay queries.
_TEXT_TRANSFORM = re.compile(
    r'^(?:(?:请|麻烦|能否|能不能|可以|帮我|替我|再|先|然后|并|也)\s*)*'
    r'(?:[把将].{1,30}?(?:缩短|精简|压缩|扩写|翻译|删减|润色|改写)|'
    r'(?:缩短|精简|压缩|扩写|翻译|删减|润色|改写)(?!过|后|前|了[吗么])|'
    r'短(?:一?点|一些)|长(?:一?点|一些))|'
    r'^(?:(?:please|can you|could you|would you|and|then)\s+)*'
    r'(?:(?:shorten|condense|expand|translate|polish|rephrase|rewrite)\b|'
    r'make\s+(?:it|this|that|the\s+\w+)\s+(?:shorter|longer|clearer|simpler)\b)', re.I)
# Explicit output targets distinguish editing from questions about stored edits.
_COMMAND_PREFIX = r'^(?:(?:请|麻烦|能否|能不能|可以|帮我|替我|再|先|然后|并|也)\s*)*'
_TEXT_TARGET = re.compile(_COMMAND_PREFIX +
    r'(?:[把将].{1,24}?)?(?:(?:缩|压|减)(?:到|成|为).{1,16}(?:字|句|行|段|条|短信|邮件)|'
    r'(?:改|调整)(?:到|成|为)\s*(?:[0-9]{1,4}|[零〇一二两三四五六七八九十百千]{1,8})'
    r'(?:个)?(?:字|句话|句|行|段|条))', re.I)
_TEXT_STYLE = re.compile(_COMMAND_PREFIX +
    r'(?:[把将])?(?:语气|口吻|措辞|文风|字数|篇幅|长度|格式)'
    r'(?:.{0,20}(?:一点|一些|些|点)|.{0,6}(?:改|调整|换|缩|压|减)(?:为|成|得|到).{1,20})|'
    + _COMMAND_PREFIX + r'保留(?:原意|原来的意思|原有条件)', re.I)
_GUIDANCE = re.compile(_COMMAND_PREFIX +
    r'(?:提醒我|提醒一下|列出|列一下).{0,30}(?:步骤|准备|注意|发(?:送)?(?:前|之前)|遗漏|没补|补什么|检查)', re.I)
# Unlike a bare occurrence of "write an email", an imperative at the clause
# start requests a new draft. Questions about earlier writing remain readbacks.
_WRITE_DRAFT = re.compile(
    r'^(?:(?:please|can you|could you|would you|and|then)\s+)*write\s+'
    r'(?:(?:me|us|a|an|the|short|brief|polite|formal)\s+){0,4}'
    r'(?:email|letter|message|reply)\b', re.I)
# Compact spoken requests still need an imperative position or an explicit
# request marker. They use the same past/readback exclusions as text edits.
_COMPACT_DRAFT = re.compile(
    r'(?:^|(?:请|麻烦|帮我|替我|能否|能不能)\s*(?:给(?:他|她|对方))?)(?:起|拟|写)(?:一)?'
    r'(?:封(?:邮件|信)|条(?:短信|消息)|段(?:话|回复))', re.I)

_NEGATED = re.compile(r'(?:不要|不用|无需|不必|别|不想|不需要).{0,8}$|'
                      r'\b(?:do not|don\x27t|no need to|without)\s+(?:\w+\s+){0,3}$', re.I)
_NEGATED_COMMAND = re.compile(r'^(?:请|麻烦)?(?:不要|不用|无需|不必|别|不想|不需要)|'
                              r'^(?:please\s+)?(?:do not|don\x27t|no need to)\b', re.I)
_TEXT_READBACK = re.compile(r'何时|哪天|什么时候|日期|时间|是(?:什么|谁|哪)|是否|有没有|'
                            r'(?:查|找)(?:出来|到)|(?:缩短|精简|压缩|扩写|翻译|删减|润色|改写)[稿件本]')
_PAST_CONTEXT = re.compile(r'(?:当时|以前|之前|曾经)|^(?:他|她|他们|她们)')
_REQUEST_PREFIX = re.compile(r'请|帮我|替我|麻烦|能否|能不能|想要')
_CLAUSES = re.compile(r'[，,。！？!?；;\n]+|但是|但|不过|而是|\bbut\b', re.I)
_GREETING = re.compile(r'\s*(?:你好|您好|嗨|哈[喽啰]|早上好|晚上好|谢谢|hello|hi|thanks)[！!。,.，\s]*', re.I)
_DRAFT_OUTPUT = re.compile(
    r'起草|草拟|拟[一几条份段个封]|拟写|代[写笔]|写(?:一|个|条|段|封|份|给)|'
    r'回复(?:一|他|她|对方)|开场白|话术|文案|措辞|'
    r'\b(?:draft|compose|rewrite)\b', re.I)

# Reading the previous recipient's sources needs a stricter boundary than
# merely permitting generation. Objects here can only point at prior output;
# arbitrary names, quoted/pasted new prose and new drafting tasks do not qualify.
_EDIT_NOUN = r'(?:邮件|短信|消息|草稿|回复|回答|正文|文字|内容)'
_EDIT_OBJECT = (
    r'(?:(?:这|那)(?:一)?(?:封|条|段|份|个)(?:' + _EDIT_NOUN + r')?|'
    r'(?:这|那)(?:个)?' + _EDIT_NOUN + r'|'
    r'(?:上|前)(?:一)?(?:封|条|段|份|个)(?:' + _EDIT_NOUN + r')?|'
    r'(?:上面|前面|上述|刚才)(?:的)?' + _EDIT_NOUN + r')')
_EDIT_TARGET = r'(?:[0-9]{1,4}|[零〇一二两三四五六七八九十百千]{1,8})(?:个)?(?:字|句话|句|行|段|条)'
_EDIT_END = r'(?:以内|以下|左右)?(?:就够了|就够|就可以了|就可以|即可|就行了|就行|就好|吧|一下)?'
_EDIT_LANGUAGE = r'(?:英文|英语|中文|汉语|日文|日语|韩文|韩语|法文|法语|德文|德语|西班牙语|俄文|俄语|阿拉伯语|葡萄牙语)'
_EDIT_STYLE = r'(?:随和|正式|自然|口语化|礼貌|客气|简洁|友好|轻松|温和|委婉)'
_STRICT_EDIT = re.compile(
    _COMMAND_PREFIX + r'(?:(?:[把将]\s*)?' + _EDIT_OBJECT + r'\s*)?'
    r'(?:(?:缩短|精简|压缩|扩写|删减|润色|改写)(?:一下|一点|一些)?|'
    r'(?:缩短|精简|压缩|删减|缩|压|减|改|调整)(?:到|成|为)' + _EDIT_TARGET + r'|'
    r'(?:短|长)(?:一点|一些)|(?:翻译|译)(?:成|为)' + _EDIT_LANGUAGE + r')' + _EDIT_END + r'$|'
    + _COMMAND_PREFIX + r'(?:[把将])?(?:语气|口吻|措辞|文风)(?:(?:改|调整|换)(?:成|为|得))?'
    + _EDIT_STYLE + r'(?:一点|一些|些|点)?(?:吧)?$', re.I)
_EDIT_MODIFIER = re.compile(
    _COMMAND_PREFIX + r'保留(?:原意|原来的意思|原有条件|原来的条件|原有顺序|原来的顺序)$|'
    r'^(?:别|不要|不用|不必|无需)(?:太)?(?:正式|生硬|冗长|客套)(?:了|吧)?$|'
    r'^(?:太(?:长|短|正式|生硬|冗长|客套)|不够(?:自然|简洁|随和|礼貌|正式|清楚)|'
    r'有点(?:长|短|生硬|正式|啰嗦))(?:了|吧)?$', re.I)
_EDIT_CHECKLIST = re.compile(
    r'^(?:(?:然后|再|并|也|最后|末尾|另外|请|帮我)\s*)*'
    r'(?:另(?:外)?列(?:出)?|列出|列一下|提醒)(?:我|一下)?(?:点)?(?:发|发送)(?:前|之前)'
    r'(?:(?:还)?(?:要|应该|需要)?(?:补齐|补充|补|准备|检查|核对|完成)(?:的)?'
    r'(?:项目|事项|信息|材料|步骤|内容)|还有什么没补|还需要补什么|还要补什么)$', re.I)


def is_text_edit_request(question):
    """Detect an anchored edit command, including edits with new text objects.

    This wider detector is only used to route an inherited pronoun through the
    stricter prior-draft boundary. It never grants generation or reading scope.
    """
    if not isinstance(question, str):
        return False
    for clause in _CLAUSES.split(question):
        clause = clause.strip()
        if _NEGATED_COMMAND.search(clause):
            continue
        if any(pattern.match(clause) for pattern in (_TEXT_TRANSFORM, _TEXT_TARGET, _TEXT_STYLE)):
            return True
    return False


def is_draft_edit_followup(question):
    """Recognize only closed edits of prior output, for recipient inheritance.

    This grants neither write permission nor a semantic guarantee. Requiring
    every clause to be an edit/modifier prevents new named text or pasted prose
    from borrowing a previous recipient's reading scope.
    """
    if not isinstance(question, str) or _TEXT_READBACK.search(question):
        return False
    clauses = [part.strip() for part in _CLAUSES.split(question) if part.strip()]
    return bool(clauses) and any(_STRICT_EDIT.fullmatch(part) for part in clauses) and all(
        _STRICT_EDIT.fullmatch(part) or _EDIT_MODIFIER.fullmatch(part) or _EDIT_CHECKLIST.fullmatch(part)
        for part in clauses)


def presentation_contract(question):
    requested = False
    draft_requested = False
    if isinstance(question, str):
        requested = bool(_GREETING.fullmatch(question))
        for clause in _CLAUSES.split(question):
            clause = clause.strip()
            matches = [match for match in _GENERATION.finditer(clause)
                       if not _PAST_CONTEXT.search(clause[:match.start()]) or
                       _REQUEST_PREFIX.search(clause[:match.start()])]
            if not _TEXT_READBACK.search(clause):
                matches += [match for pattern in (_TEXT_TRANSFORM, _TEXT_TARGET, _TEXT_STYLE, _GUIDANCE, _WRITE_DRAFT,
                                                  _COMPACT_DRAFT)
                            for match in pattern.finditer(clause)]
                # "She helped me write..." describes a past event; an explicit
                # command at the start remains a request.
                matches = [match for match in matches
                           if not _PAST_CONTEXT.search(clause[:match.start()]) or
                           _REQUEST_PREFIX.search(clause[:match.start()])]
            if not _NEGATED_COMMAND.search(clause) and any(
                    not _NEGATED.search(clause[:match.start()]) for match in matches):
                requested = True
                draft_requested = draft_requested or any(
                    not _NEGATED.search(clause[:match.start()])
                    for pattern in (_DRAFT_OUTPUT, _TEXT_TRANSFORM, _TEXT_TARGET, _TEXT_STYLE, _WRITE_DRAFT,
                                    _COMPACT_DRAFT)
                    for match in pattern.finditer(clause))
    return {'version': 2, 'facts': 'server_records', 'generation_requested': requested,
            'draft_requested': draft_requested,
            'method': 'conservative_language_heuristic', 'semantic_verified': False}
