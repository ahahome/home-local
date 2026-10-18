ALTER TABLE memory_item ADD COLUMN status TEXT NOT NULL DEFAULT 'active'
  CHECK(status IN ('active','withdrawn'));
ALTER TABLE memory_item ADD COLUMN revision INTEGER NOT NULL DEFAULT 0 CHECK(revision>=0);
ALTER TABLE memory_item ADD COLUMN assignment_revision INTEGER NOT NULL DEFAULT 0
  CHECK(assignment_revision>=0);
ALTER TABLE memory_item ADD COLUMN updated_at INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memory_item ADD COLUMN origin_evidence_id TEXT;
UPDATE memory_item SET updated_at=created_at,origin_evidence_id=evidence_id;

CREATE TABLE memory_correction (
  id TEXT PRIMARY KEY,
  memory_id TEXT NOT NULL REFERENCES memory_item(id) ON DELETE CASCADE,
  action TEXT NOT NULL CHECK(action IN ('edit','withdraw','restore')),
  before_json TEXT NOT NULL,
  after_json TEXT NOT NULL,
  reason TEXT NOT NULL DEFAULT '',
  request_id TEXT NOT NULL,
  request_hash TEXT NOT NULL,
  revision INTEGER NOT NULL,
  evidence_id TEXT,
  created_at INTEGER NOT NULL,
  UNIQUE(memory_id,request_id)
);
CREATE INDEX idx_memory_correction_history ON memory_correction(memory_id,revision DESC);
CREATE INDEX idx_memory_status ON memory_item(status,ts DESC) WHERE deleted_at IS NULL;
