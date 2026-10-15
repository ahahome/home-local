"""Shared identity and event rules for tools, extraction and manual confirmation."""
from __future__ import annotations
from decimal import Decimal
from datetime import datetime
import re

from app.store import repo
from app.util import sha256_text, today_local


class AmbiguousPerson(ValueError):
    def __init__(self, name, candidates):
        super().__init__(f"「{name}」对应多个人，请先选择人物")
        self.candidates = [{"person_id": p["id"], "display_name": p["display_name"]}
                           for p in candidates]


def discard_key(kind, fingerprint, source_id):
    return sha256_text(f'discard|{kind}|{fingerprint}|{source_id}')


def resolve_person(conn, name=None, person_id=None, create=False):
    if person_id:
        p = repo.get_person(conn, person_id)
        if not p or p["deleted_at"] or p["status"] == "merged":
            raise ValueError("人物不存在或已归档")
        return person_id
    if not name:
        return None
    rows = conn.execute(
        "SELECT DISTINCT p.* FROM person p JOIN person_alias a ON a.person_id=p.id "
        "WHERE a.alias=? AND p.deleted_at IS NULL AND p.status!='merged'", (name,)).fetchall()
    if len(rows) > 1:
        raise AmbiguousPerson(name, rows)
    if rows:
        return rows[0]["id"]
    if create:
        from app.memory.extract import is_role_person
        if is_role_person(name):
            raise ValueError("角色词不能作为人物建档")
        return repo.create_person(conn, name, status="active")
    raise ValueError(f"没有找到「{name}」的档案")


def ledger_key(person_id, direction, cents, day=None):
    return sha256_text(f"ledger|{person_id}|{direction}|{cents}|{day or today_local()}")


def promise_key(person_id, who, content, due=None, recipient_names=()):
    from app.util import normalize_value
    if due:
        dates = [re.escape(due)]
        try:
            parsed = datetime.strptime(due, '%Y-%m-%d')
            dates += [rf'{parsed.year}年0?{parsed.month}月0?{parsed.day}日', rf'0?{parsed.month}月0?{parsed.day}日']
        except ValueError:
            pass
        content = re.sub(r'^(?:在|于)?(?:' + '|'.join(dates) + r')(?:之前|以前|前)?[\s，,：:]*', '', content.strip())
    if who == 'me' and recipient_names:
        names = '|'.join(re.escape(name) for name in sorted(set(recipient_names), key=len, reverse=True) if name)
        if names:
            content = re.sub(r'^(?:给|向)\s*(?:' + names + r')\s*', '', content.strip())
    return sha256_text(f"promise|{person_id}|{who}|{normalize_value(content)}|{due}")


def matching_promise(conn, pid, who, content, due, due_ms):
    """Recognize existing rows too, including keys written before date normalization."""
    # The recipient is already stored in person_id; tools can omit it from the action.
    names = repo.aliases_of(conn, pid) if pid else []
    person = repo.get_person(conn, pid) if pid else None
    if person:
        names = [*names, person['display_name']]
    key = promise_key(pid, who, content, due, names)
    for row in conn.execute('SELECT id,content FROM promise WHERE person_id IS ? AND who=? AND due_at IS ?',
                            (pid, who, due_ms)):
        if promise_key(pid, who, row['content'], due, names) == key:
            return row['id']
    return None


def explicit_transfer_direction(source, names, cents):
    """Only override a contradiction when named parties and this exact amount are explicit.

    Multiple matching directions, indirect speech and unsupported phrasing stay with
    the ordinary confidence gate. This is not a general financial-language parser.
    """
    directions = set()
    amount_pattern = re.compile(r'(\d+(?:\.\d{1,2})?)\s*(?:元|块|CNY)')
    verbs = r'(?:转账|转了|转|付了|支付了|支付|打款|打了|汇款|汇了|还了|还)'
    adverb = r'(?:已经|已|刚刚|刚|今天|昨天)?'
    for clause in re.split(r'[。；;\n]', source):
        speaker = re.match(r'^\s*(?:\d{1,2}:\d{2}\s+)?([\u4e00-\u9fffA-Za-z]{1,20})[:：]\s*(?=[^\s\d])', clause)
        if speaker and speaker[1] not in {'我', '本人', '用户', 'me', 'Me'}:
            continue
        for amount in amount_pattern.finditer(clause):
            if int(Decimal(amount[1]) * 100) != cents:
                continue
            prefix = re.sub(r'\s+', '', clause[max(0, amount.start()-80):amount.start()])
            for name in set(names):
                if not name:
                    continue
                person = re.escape(name)
                out = rf'我{adverb}(?:给|向|转给|付给){person}{verbs}[^\d]{{0,12}}$'
                incoming = rf'{person}{adverb}(?:给|向|转给|付给)我{verbs}[^\d]{{0,12}}$'
                receipt = rf'我(?:收到了|收到){person}(?:转账|转|还款|付款)[^\d]{{0,12}}$'
                if re.search(out, prefix):
                    directions.add('out')
                if re.search(incoming, prefix) or re.search(receipt, prefix):
                    directions.add('in')
    return next(iter(directions)) if len(directions) == 1 else None
