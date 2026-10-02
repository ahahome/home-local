-- Manual corrections preserve original evidence and before/after snapshots.
ALTER TABLE ledger_entry ADD COLUMN revision INTEGER NOT NULL DEFAULT 0;
CREATE TABLE correction_log (
    id TEXT PRIMARY KEY,
    entity TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    person_id TEXT,
    related_person_id TEXT,
    field TEXT,
    action TEXT NOT NULL,
    before_json TEXT NOT NULL,
    after_json TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL
);
CREATE INDEX idx_correction_person ON correction_log(person_id, created_at);
CREATE INDEX idx_correction_entity ON correction_log(entity, entity_id, created_at);
