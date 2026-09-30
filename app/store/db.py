"""SQLite 连接与迁移：WAL、外键、busy_timeout、user_version 迁移、savepoint。"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from app.config import get_settings

SCHEMA_VERSION = 1
_local = threading.local()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def get_conn() -> sqlite3.Connection:
    """每线程一个连接。写入用单连接 + savepoint 保证顺序。"""
    conn = getattr(_local, "conn", None)
    if conn is None:
        s = get_settings()
        s.abs_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = _connect(s.abs_db_path)
        migrate(conn)
        _local.conn = conn
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    """user_version 迁移。v1 = schema.sql 全量。后续版本用 migrations/00N_*.sql。"""
    (ver,) = conn.execute("PRAGMA user_version").fetchone()
    if ver < 1:
        schema = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
        conn.executescript(schema)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.commit()
        ver = SCHEMA_VERSION

    mig_dir = Path(__file__).resolve().parent.parent.parent / "migrations"
    if mig_dir.is_dir():
        for f in sorted(mig_dir.glob("*.sql")):
            try:
                n = int(f.stem.split("_")[0])
            except ValueError:
                continue
            if n > ver:
                conn.executescript(f.read_text(encoding="utf-8"))
                conn.execute(f"PRAGMA user_version={n}")
                conn.commit()
                ver = n


@contextmanager
def savepoint(conn: sqlite3.Connection, name: str):
    """SAVEPOINT 上下文：异常只回滚到保存点，不影响外层事务中已成功的部分。"""
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
    except Exception:
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
        raise
    else:
        conn.execute(f"RELEASE {name}")


def reset_for_tests(db_path: Path):
    """测试用：关掉线程局部连接。"""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
