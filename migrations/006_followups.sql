CREATE TABLE followup (
  id TEXT PRIMARY KEY,
  person_id TEXT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
  origin_person_id TEXT NOT NULL,
  goal TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'preparing'
    CHECK(status IN ('preparing','ready','awaiting_reply','follow_up','completed')),
  brief TEXT,
  draft TEXT NOT NULL DEFAULT '',
  draft_source_ids TEXT NOT NULL DEFAULT '[]',
  task_id TEXT REFERENCES task(id) ON DELETE SET NULL,
  revision INTEGER NOT NULL DEFAULT 1,
  request_id TEXT NOT NULL,
  request_hash TEXT NOT NULL,
  prepare_status TEXT NOT NULL DEFAULT 'queued',
  reminder_id TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE(origin_person_id, request_id)
);
CREATE INDEX idx_followup_person ON followup(person_id, created_at DESC);

CREATE TABLE followup_outcome (
  id TEXT PRIMARY KEY,
  followup_id TEXT NOT NULL REFERENCES followup(id) ON DELETE CASCADE,
  request_id TEXT NOT NULL,
  request_hash TEXT NOT NULL,
  payload TEXT NOT NULL,
  contact_id TEXT NOT NULL,
  reminder_id TEXT,
  receipts TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  UNIQUE(followup_id, request_id)
);
CREATE INDEX idx_followup_outcome ON followup_outcome(followup_id, created_at);
