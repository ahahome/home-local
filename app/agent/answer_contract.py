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
    r'(?:[把将].{1,24}?)?(?:缩|压|减)(?:到|成|为).{1,16}(?:字|句|行|段|条|短信|邮件)', re.I)
_TEXT_STYLE = re.compile(_COMMAND_PREFIX +
    r'(?:[把将])?(?:语气|口吻|措辞|文风|字数|篇幅|长度|格式)'
    r'(?:.{0,20}(?:一点|一些|些|点)|.{0,6}(?:改|调整|换|缩|压|减)(?:为|成|得|到).{1,20})|'
    + _COMMAND_PREFIX + r'保留(?:原意|原来的意思|原有条件)', re.I)
_GUIDANCE = re.compile(_COMMAND_PREFIX +
    r'(?:提醒我|提醒一下|列出|列一下).{0,30}(?:步骤|准备|注意|发(?:送)?(?:前|之前)|遗漏|没补|补什么|检查)', re.I)

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


def presentation_contract(question):
    requested = False
    if isinstance(question, str):
        requested = bool(_GREETING.fullmatch(question))
        for clause in _CLAUSES.split(question):
            clause = clause.strip()
            matches = [match for match in _GENERATION.finditer(clause)
                       if not _PAST_CONTEXT.search(clause[:match.start()]) or
                       _REQUEST_PREFIX.search(clause[:match.start()])]
            if not _TEXT_READBACK.search(clause):
                matches += [match for pattern in (_TEXT_TRANSFORM, _TEXT_TARGET, _TEXT_STYLE, _GUIDANCE)
                            for match in pattern.finditer(clause)]
            if not _NEGATED_COMMAND.search(clause) and any(
                    not _NEGATED.search(clause[:match.start()]) for match in matches):
                requested = True
                break
    return {'version': 1, 'facts': 'server_records', 'generation_requested': requested,
            'method': 'conservative_language_heuristic', 'semantic_verified': False}
