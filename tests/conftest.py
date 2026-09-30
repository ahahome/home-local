import pytest


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    """每个测试一个独立的临时数据库，且不触网（LLM key 置空，相关调用走 try/except 兜底）。"""
    monkeypatch.setenv("HOME_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("HOME_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("HOME_LLM_API_KEY", "")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.store import db
    db.reset_for_tests(tmp_path / "test.db")
    c = db.get_conn()
    yield c
    db.reset_for_tests(tmp_path / "test.db")
    get_settings.cache_clear()
