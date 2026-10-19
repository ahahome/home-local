# Portable Demo Data

This directory contains the synthetic data used in the local QA workspace,
updated on 2026-10-19 from `data/qa/memory-quality-v08-20261019/preview/`.
All people, conversations, commitments, and transfers are fictional.

The committed snapshot includes:

- `home.db`: a standalone SQLite backup, including committed WAL contents.
- `blobs/`: the three original attachments, stored by SHA-256.
- 70 people (65 visible), 218 field versions, 346 interactions, 126 memories,
  69 commitments, 131 ledger entries, 32 confirmation records, 3 projects,
  21 completed tasks, and 6 conversations with 24 messages. The database uses
  schema version 9.
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

The snapshot retains every row from the earlier demo, including the exact older
conversation content and source metadata. A fresh restore passed SQLite integrity
and foreign-key checks, attachment hash checks, and saved citation/evidence checks.

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
