# Portable Demo Data

This directory contains the synthetic data used in the local QA workspace,
updated on 2026-10-19 from `data/qa/answer-stability-v010-20261019/preview/`.
All people, conversations, commitments, and transfers are fictional.

The committed snapshot includes:

- `home.db`: a standalone SQLite backup, including committed WAL contents.
- `blobs/`: the three original attachments, stored by SHA-256.
- 71 people (66 visible), 220 field versions, 347 interactions, 126 memories,
  69 commitments, 131 ledger entries, 32 confirmation records, 3 projects,
  30 tasks (29 completed and one partial), and 9 conversations with 47 stored
  messages (including five tool messages). The database uses schema version 9.
- Three synthetic communication followups with four saved outcomes and four
  memory analyses, including five confirmed proposals and one discarded proposal.
  These demonstrate historical field updates, communication preferences, source
  review, and outcomes with no new information. There are no pending memory-update
  proposals in this snapshot; select all statuses in the memory proposal view to
  inspect the processing history.
- A fictional person named `沈知遥` and the conversation `v0.7 来源可追溯演示`
  demonstrate dated city fields and communication preferences. Two real gateway
  answers retain their original source snapshots: the earlier answer cites the
  original Wednesday preference, while the newer answer cites the manually
  corrected Thursday preference. Opening the older citation reports that the
  record has changed; the correction history and original evidence remain.
- Two additional fictional communication preferences distinguish the two people
  named `张晨`. The conversations `v0.8 同名选择与指代演示` and
  `v0.8 多人范围澄清演示` preserve clickable identity choices, a pronoun followup,
  a comparison, and clarification after an ambiguous pronoun. The real gateway
  answers remain unchanged, including an uncited followup and its warning;
  these are examples for inspection, not a claim that every answer is complete.
- Two v0.9 conversations retain six real gateway turns as product demonstrations,
  rather than a benchmark. `v0.9 可信回答与写入边界演示` contains four observations
  made before the final write guards: a cited communication-preference query,
  a blood-type query with the fixed insufficient-information answer and no
  uncited warning, and two mixed inputs that safely fell back to read-only
  processing when the intent classifier was unavailable. All four original
  answers and metadata remain unchanged.
- `v0.9 最终版混合写入演示 · 648e46e7` contains two turns from the final application
  version. An input explicitly labelled as fictional demonstration material
  remains read-only under the non-assertive policy. A subsequent natural factual
  statement saves the fictional person `顾予澄`, company `澄川设计`, and city
  `厦门`, while answering the separate question about 沈知遥 with citation S3.
  The saved note contains only the factual statement; the complete input stays
  in its original audit evidence. No new memory or confirmation records were
  created by these six turns. The extra natural statement corrects the QA input
  so the read-only hypothetical case is not mistaken for a normal mixed-input
  failure. Original wording issues, including a repeated `建议：`, are retained.
- `0.10 记忆日期追问` retains a real gateway contact-preference query followed
  by a deterministic date question. The new turn re-reads the same memory and
  distinguishes its 2026-09-12 labelled date from its 2026-10-18 initial capture
  and audited manual edit dates, with a fresh citation S1 and no model request
  for the date answer. A separate explicit reanalysis of a failed v0.9 message
  adds a linked user message and answer in that original conversation. The new
  classifier reads its explicit fictional-example qualifier and keeps it
  read-only; duplicate HTTP requests return the same task. All older messages
  and source metadata remain unchanged. These three new turns are product
  demonstrations, separate from the 47 formal evaluation answers.

The snapshot retains all 2,627 application rows from the v0.9 demo when comparing
without conversation update timestamps and reminder firing state. All 41 older
messages retain their exact content and metadata. During preview, three existing
overdue reminders fired normally; they were not added by a chat turn. A fresh
restore passed SQLite integrity and foreign-key checks, three attachment hash
checks, 410 checked evidence-field references and 15 saved citation snapshots.
Desktop and narrow-screen screenshots are available in `docs/assets/v010/`;
other QA artifacts remain under the ignored local data folder.

After installing `requirements.txt`, restore a working copy from the repository
root:

```sh
python -m scripts.restore_demo
```

The command uses `HOME_DB_PATH` and `HOME_BLOB_DIR` from the environment or `.env`,
defaulting to `data/home.db` and `data/blobs/`. It validates attachment hashes and
refuses to overwrite an existing database or a conflicting attachment. To keep an
existing database, select new paths through those settings and use the same
settings when starting the service.

Run the application against the restored copy. Keep this directory as a fixed
snapshot; local changes under `data/` remain ignored by Git. Runtime logs,
browser traces, API keys, and SQLite WAL/SHM files are not part of the snapshot.
