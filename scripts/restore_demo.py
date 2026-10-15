"""Restore the committed demo snapshot without overwriting local data."""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

from app.config import ROOT, get_settings

DEMO_DIR = ROOT / "demo"


def restore_demo(db_path: Path, blob_dir: Path, *, source_dir: Path = DEMO_DIR) -> int:
    if db_path.exists():
        raise FileExistsError(f"Database already exists; choose a new HOME_DB_PATH: {db_path}")

    source_db = (source_dir / "home.db").resolve()
    with closing(sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True)) as source:
        if source.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Demo database failed its integrity check")
        blobs = [path for path in sorted((source_dir / "blobs").glob("*/*")) if path.is_file()]
        for path in blobs:
            if path.parent.name != path.name[:2] or hashlib.sha256(path.read_bytes()).hexdigest() != path.name:
                raise ValueError(f"Invalid demo attachment: {path.name}")
            dest = blob_dir / path.parent.name / path.name
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != path.name:
                raise ValueError(f"Existing attachment has different contents: {dest}")
        references = {row[0] for row in source.execute(
            "SELECT DISTINCT blob_hash FROM evidence WHERE blob_hash IS NOT NULL")}
        if references - {path.name for path in blobs}:
            raise ValueError("Demo snapshot is missing referenced attachments")

        for path in blobs:
            dest = blob_dir / path.parent.name / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                with dest.open("xb") as output:
                    output.write(path.read_bytes())

        # Reserve the destination exclusively before SQLite opens it.
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with db_path.open("xb"):
            pass
        try:
            with closing(sqlite3.connect(db_path)) as target:
                source.backup(target)
                target.execute("PRAGMA journal_mode=DELETE")
        except Exception:
            db_path.unlink(missing_ok=True)
            raise
    return len(blobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    settings = get_settings()
    try:
        count = restore_demo(settings.abs_db_path, settings.abs_blob_dir)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f"Cannot restore demo: {exc}\n")
    print(f"Restored demo database: {settings.abs_db_path}")
    print(f"Attachments: {count} files in {settings.abs_blob_dir}")


if __name__ == "__main__":
    main()
