-- Historical records may be uploads, queries or drafts; none prove contact.
ALTER TABLE interaction ADD COLUMN kind TEXT NOT NULL DEFAULT 'note'
CHECK (kind IN ('note', 'contact'));

CREATE INDEX IF NOT EXISTS idx_interaction_kind_ts ON interaction(kind, ts DESC);
CREATE INDEX IF NOT EXISTS idx_ip_participant ON interaction_person(person_id, interaction_id)
WHERE role='participant';

-- Previously calculated importance included notes and cannot be reused.
UPDATE person SET importance=0.0;
