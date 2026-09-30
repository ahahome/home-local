-- home 数据库 schema v1
-- 设计要点：
--  person_field 双时态版本化（不覆盖只追加）
--  合并保留墓碑 + merge_log 支持反向拆分
--  interaction/ledger/applied_patch 均有幂等键
--  删除用 deleted_at 软删，且删除后不得复活（写入侧检查）

CREATE TABLE IF NOT EXISTS person (
  id            TEXT PRIMARY KEY,
  display_name  TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'unconfirmed',  -- unconfirmed | active | merged
  merged_into   TEXT,
  value_read    TEXT,
  value_read_evidence TEXT,
  importance    REAL NOT NULL DEFAULT 0.5,
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL,
  deleted_at    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_person_status ON person(status) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS person_alias (
  person_id   TEXT NOT NULL REFERENCES person(id),
  alias       TEXT NOT NULL,
  source      TEXT NOT NULL,            -- chat | contact | ocr | crawler | manual
  confidence  REAL NOT NULL DEFAULT 1.0,
  created_at  INTEGER NOT NULL,
  -- 注意：alias 不能全局唯一 —— 多个人物可以共享同一个别名（如三个"小张"），
  -- 这正是身份消歧要处理的场景。唯一性按 (person_id, alias, source)。
  PRIMARY KEY (person_id, alias, source)
);
CREATE INDEX IF NOT EXISTS idx_alias_lookup ON person_alias(alias);
CREATE INDEX IF NOT EXISTS idx_alias_person ON person_alias(person_id);

CREATE TABLE IF NOT EXISTS person_field (
  person_id   TEXT NOT NULL,
  field       TEXT NOT NULL,            -- company | title | city | birthday | trait.* | need.*
  value       TEXT,
  confidence  REAL NOT NULL,
  evidence_id TEXT,
  valid_from  INTEGER NOT NULL,         -- 业务时间
  valid_to    INTEGER,                  -- NULL = 当前有效
  recorded_at INTEGER NOT NULL,         -- 系统时间
  PRIMARY KEY (person_id, field, valid_from)
);
CREATE INDEX IF NOT EXISTS idx_field_current ON person_field(person_id, field) WHERE valid_to IS NULL;

CREATE TABLE IF NOT EXISTS interaction (
  id            TEXT PRIMARY KEY,
  ts            INTEGER NOT NULL,
  channel       TEXT NOT NULL DEFAULT 'chat',   -- chat | screenshot | voice | web | import
  source_ref    TEXT,
  content       TEXT NOT NULL,
  content_hash  TEXT NOT NULL UNIQUE,   -- 幂等：重放同一条内容不重复入库
  created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_interaction_ts ON interaction(ts);

CREATE TABLE IF NOT EXISTS interaction_person (
  interaction_id TEXT NOT NULL REFERENCES interaction(id),
  person_id      TEXT NOT NULL,
  role           TEXT DEFAULT 'mentioned',   -- mentioned | from | to
  PRIMARY KEY (interaction_id, person_id)
);
CREATE INDEX IF NOT EXISTS idx_ip_person ON interaction_person(person_id);

CREATE TABLE IF NOT EXISTS memory_item (
  id          TEXT PRIMARY KEY,
  person_id   TEXT,                     -- NULL = 关于用户自己 / 全局
  kind        TEXT NOT NULL,            -- semantic | episodic | reflective | journal | skill
  text        TEXT NOT NULL,
  confidence  REAL NOT NULL DEFAULT 1.0,
  evidence_id TEXT,
  ts          INTEGER NOT NULL,
  created_at  INTEGER NOT NULL,
  deleted_at  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_memory_person ON memory_item(person_id, kind) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory_item(kind) WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS relation_edge (
  id          TEXT PRIMARY KEY,
  from_person TEXT NOT NULL,
  to_person   TEXT NOT NULL,
  kind        TEXT NOT NULL,            -- referred_by | colleague | family | same_group | know
  strength    REAL NOT NULL DEFAULT 0.5,
  evidence_id TEXT,
  created_at  INTEGER NOT NULL,
  UNIQUE (from_person, to_person, kind)
);

CREATE TABLE IF NOT EXISTS promise (
  id          TEXT PRIMARY KEY,
  person_id   TEXT,
  who         TEXT NOT NULL,            -- me | them
  content     TEXT NOT NULL,
  due_at      INTEGER,
  status      TEXT NOT NULL DEFAULT 'open',   -- open | done | cancelled | expired
  evidence_id TEXT,
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_promise_status ON promise(status, due_at);

CREATE TABLE IF NOT EXISTS ledger_entry (
  id           TEXT PRIMARY KEY,
  person_id    TEXT,
  direction    TEXT NOT NULL,           -- in | out
  amount_cents INTEGER NOT NULL,
  currency     TEXT NOT NULL DEFAULT 'CNY',
  note         TEXT,
  occurred_at  INTEGER NOT NULL,
  status       TEXT NOT NULL DEFAULT 'confirmed',  -- confirmed | pending | void
  dedupe_hash  TEXT NOT NULL UNIQUE,
  evidence_id  TEXT,
  created_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_person ON ledger_entry(person_id) WHERE status != 'void';

CREATE TABLE IF NOT EXISTS skill (
  id           TEXT PRIMARY KEY,
  name         TEXT NOT NULL UNIQUE,
  person_id    TEXT,
  trigger_hint TEXT,
  style_notes  TEXT,
  body         TEXT NOT NULL,
  created_at   INTEGER NOT NULL,
  updated_at   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
  id         TEXT PRIMARY KEY,
  kind       TEXT NOT NULL,             -- blob | text | url
  blob_hash  TEXT,
  source_ref TEXT,
  excerpt    TEXT,
  created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_job (
  id              TEXT PRIMARY KEY,
  kind            TEXT NOT NULL,        -- message | screenshot | document | webpage | import
  status          TEXT NOT NULL,        -- queued | parsing | done | partial | failed
  payload         TEXT,
  result          TEXT,
  error           TEXT,
  attempts        INTEGER NOT NULL DEFAULT 0,
  idempotency_key TEXT UNIQUE,
  created_at      INTEGER NOT NULL,
  updated_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS merge_log (
  id          TEXT PRIMARY KEY,
  survivor_id TEXT NOT NULL,
  merged_id   TEXT NOT NULL,
  snapshot    TEXT NOT NULL,            -- 被合并方全部行数据的 JSON，用于拆分
  undone      INTEGER NOT NULL DEFAULT 0,
  created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS applied_patch (
  fingerprint  TEXT PRIMARY KEY,
  person_id    TEXT,
  field        TEXT,
  value_norm   TEXT,
  source_msg_id TEXT,
  applied_at   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_item (
  id          TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,            -- person_identity | field | ledger | promise
  payload     TEXT NOT NULL,            -- JSON
  reason      TEXT,
  created_at  INTEGER NOT NULL,
  resolved_at INTEGER,
  resolution  TEXT
);
CREATE INDEX IF NOT EXISTS idx_pending_open ON pending_item(kind) WHERE resolved_at IS NULL;

CREATE TABLE IF NOT EXISTS journal (
  id         TEXT PRIMARY KEY,
  day        TEXT NOT NULL UNIQUE,      -- YYYY-MM-DD（本地时区）
  text       TEXT NOT NULL,
  created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation (
  id         TEXT PRIMARY KEY,
  title      TEXT NOT NULL DEFAULT '新对话',
  project_id TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS message (
  id              TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL REFERENCES conversation(id),
  role            TEXT NOT NULL,        -- user | assistant | system | tool
  content         TEXT,
  tool_calls      TEXT,
  created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_message_conv ON message(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS reminder (
  id         TEXT PRIMARY KEY,
  person_id  TEXT,
  promise_id TEXT,
  fire_at    INTEGER NOT NULL,
  text       TEXT NOT NULL,
  status     TEXT NOT NULL DEFAULT 'pending',   -- pending | fired | cancelled
  fired_at   INTEGER,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reminder_due ON reminder(status, fire_at);

CREATE TABLE IF NOT EXISTS notification (
  id         TEXT PRIMARY KEY,
  kind       TEXT NOT NULL,             -- reminder | system
  text       TEXT NOT NULL,
  read       INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS research (
  id         TEXT PRIMARY KEY,
  topic      TEXT NOT NULL,
  status     TEXT NOT NULL,             -- proposed | running | draft | done | failed
  proposal   TEXT,
  report_md  TEXT,
  sources    TEXT,
  draft      TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS vector_item (
  id         TEXT PRIMARY KEY,
  owner_type TEXT NOT NULL,             -- memory | interaction | person_name
  owner_id   TEXT NOT NULL,
  text       TEXT NOT NULL,
  dim        INTEGER NOT NULL,
  vector     BLOB NOT NULL,
  UNIQUE (owner_type, owner_id)
);
CREATE INDEX IF NOT EXISTS idx_vector_owner ON vector_item(owner_type);

CREATE TABLE IF NOT EXISTS kv (
  key   TEXT PRIMARY KEY,
  value TEXT
);

-- 全文检索：tokens 列存 jieba 预分词（空格连接）后的文本
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
  tokens,
  owner_type UNINDEXED,
  owner_id UNINDEXED,
  tokenize = 'unicode61'
);
