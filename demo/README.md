# Portable Demo Data

This directory contains the synthetic data used in the local QA workspace,
updated on 2026-10-17 from `data/qa/outcome-memory-v05-20261017/`.
All people, conversations, commitments, and transfers are fictional.

The committed snapshot includes:

- `home.db`: a standalone SQLite backup, including committed WAL contents.
- `blobs/`: the three original attachments, stored by SHA-256.
- 69 people (64 visible), 216 field versions, 346 interactions, 123 memories,
  69 commitments, 131 ledger entries, 32 confirmation records, 3 projects,
  and 14 completed tasks.
- Three synthetic communication followups with four saved outcomes and four
  memory analyses, including five confirmed proposals and one discarded proposal.
  These demonstrate historical field updates, communication preferences, source
  review, and outcomes with no new information. There are no pending memory-update
  proposals in this snapshot; select all statuses in the memory proposal view to
  inspect the processing history.

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
