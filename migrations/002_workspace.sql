CREATE TABLE IF NOT EXISTS task (
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL,
 payload TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
 result TEXT, error TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
 idempotency_key TEXT UNIQUE, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS task_event (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL REFERENCES task(id) ON DELETE CASCADE,
 kind TEXT NOT NULL, data TEXT NOT NULL, created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_event ON task_event(task_id,seq);
CREATE TABLE IF NOT EXISTS project (
 id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
 created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS project_person (
 project_id TEXT REFERENCES project(id) ON DELETE CASCADE,
 person_id TEXT REFERENCES person(id) ON DELETE CASCADE,
 PRIMARY KEY(project_id,person_id)
);
INSERT INTO pending_item(id,kind,payload,reason,created_at)
SELECT lower(hex(randomblob(16))), 'ledger',
 json_object('ledger_id',l.id,'person_id',l.person_id,'direction',l.direction,
             'amount_cents',l.amount_cents,'note',l.note,'dedupe_hash',l.dedupe_hash),
 '历史待确认账目，不计入总额',l.created_at
FROM ledger_entry l WHERE l.status='pending' AND NOT EXISTS(
 SELECT 1 FROM pending_item p WHERE p.kind='ledger' AND p.resolved_at IS NULL
 AND json_extract(p.payload,'$.dedupe_hash')=l.dedupe_hash
);
