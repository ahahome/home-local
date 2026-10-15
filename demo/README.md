# Portable Demo Data

This directory contains the synthetic data used in the local QA workspace,
exported on 2026-10-15 from `data/qa/mock-replay-20261013-162624/`.
All people, conversations, commitments, and transfers are fictional.

The committed snapshot includes:

- `home.db`: a standalone SQLite backup, including committed WAL contents.
- `blobs/`: the three original attachments, stored by SHA-256.
- 66 people (61 visible), 208 field versions, 342 interactions, 121 memories,
  69 commitments, 131 ledger entries, 26 confirmation records, 3 projects,
  and 7 completed tasks.

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
