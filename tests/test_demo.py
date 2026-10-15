"""Portable demo restoration preserves records and attachment references."""
import hashlib
import sqlite3

import pytest

from scripts.restore_demo import restore_demo


@pytest.fixture()
def snapshot(tmp_path):
    root = tmp_path / "snapshot"
    data = b"synthetic demo attachment"
    digest = hashlib.sha256(data).hexdigest()
    attachment = root / "blobs" / digest[:2] / digest
    attachment.parent.mkdir(parents=True)
    attachment.write_bytes(data)
    with sqlite3.connect(root / "home.db") as conn:
        conn.execute("CREATE TABLE evidence (blob_hash TEXT)")
        conn.execute("INSERT INTO evidence VALUES (?)", (digest,))
        conn.execute("PRAGMA user_version=4")
    return root, attachment, data


def test_restore_demo_preserves_database_and_attachments(tmp_path, snapshot):
    source, attachment, data = snapshot
    dest_db = tmp_path / "local" / "home.db"
    dest_blobs = tmp_path / "attachments"
    assert restore_demo(dest_db, dest_blobs, source_dir=source) == 1
    with sqlite3.connect(dest_db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        assert conn.execute("SELECT blob_hash FROM evidence").fetchone()[0] == attachment.name
    assert (dest_blobs / attachment.parent.name / attachment.name).read_bytes() == data


def test_restore_demo_refuses_to_overwrite_database(tmp_path, snapshot):
    source, _, _ = snapshot
    dest_db = tmp_path / "existing.db"
    dest_db.write_bytes(b"existing user data")
    dest_blobs = tmp_path / "attachments"
    with pytest.raises(FileExistsError):
        restore_demo(dest_db, dest_blobs, source_dir=source)
    assert dest_db.read_bytes() == b"existing user data"
    assert not dest_blobs.exists()


@pytest.mark.parametrize("missing", [False, True])
def test_restore_demo_rejects_incomplete_or_corrupt_attachments(tmp_path, snapshot, missing):
    source, attachment, _ = snapshot
    if missing:
        attachment.unlink()
    else:
        attachment.write_bytes(b"corrupt")
    dest_db = tmp_path / "local" / "home.db"
    dest_blobs = tmp_path / "attachments"
    with pytest.raises(ValueError):
        restore_demo(dest_db, dest_blobs, source_dir=source)
    assert not dest_db.exists()
    assert not dest_blobs.exists()


def test_restore_demo_preserves_conflicting_existing_attachment(tmp_path, snapshot):
    source, attachment, _ = snapshot
    dest_db = tmp_path / "local" / "home.db"
    dest_blobs = tmp_path / "attachments"
    existing = dest_blobs / attachment.parent.name / attachment.name
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing user attachment")
    with pytest.raises(ValueError):
        restore_demo(dest_db, dest_blobs, source_dir=source)
    assert not dest_db.exists()
    assert existing.read_bytes() == b"existing user attachment"
