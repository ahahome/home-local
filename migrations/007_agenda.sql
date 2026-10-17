CREATE TABLE contact_policy (
  person_id TEXT PRIMARY KEY REFERENCES person(id) ON DELETE CASCADE,
  enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0,1)),
  interval_days INTEGER NOT NULL DEFAULT 14 CHECK(interval_days BETWEEN 1 AND 365),
  revision INTEGER NOT NULL DEFAULT 1,
  updated_at INTEGER NOT NULL
);

CREATE TABLE agenda_action (
  person_id TEXT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
  item_id TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('snoozed','dismissed')),
  until INTEGER,
  updated_at INTEGER NOT NULL,
  PRIMARY KEY(item_id, fingerprint)
);
CREATE INDEX idx_agenda_action_person ON agenda_action(person_id);
