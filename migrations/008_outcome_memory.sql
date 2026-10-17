CREATE TABLE outcome_memory_analysis (
  outcome_id TEXT PRIMARY KEY REFERENCES followup_outcome(id) ON DELETE CASCADE,
  followup_id TEXT NOT NULL REFERENCES followup(id) ON DELETE CASCADE,
  person_id TEXT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
  task_id TEXT REFERENCES task(id) ON DELETE SET NULL,
  evidence_id TEXT,
  source_hash TEXT NOT NULL,
  analyzed_at INTEGER,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX idx_outcome_memory_person ON outcome_memory_analysis(person_id, created_at);
CREATE INDEX idx_outcome_memory_followup ON outcome_memory_analysis(followup_id, created_at);
