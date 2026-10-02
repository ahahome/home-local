"""全局配置：从 .env / 环境变量读取。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), env_prefix="HOME_", extra="ignore")

    llm_base_url: str = "https://token-hub.ailyn.com.cn/v1"
    llm_api_key: str = ""

    model_mini: str = "qwen3.7-flash"   # 日志、轻量归纳
    model_mid: str = "deepseek-v4-flash"  # 抽取、工具调用
    model_think: str = "deepseek-v4-pro"  # 分析：消歧复核、决策
    model_pro: str = "qwen3.8-max"      # 深度：Deep Research 合成
    model_vision: str = "qwen3.8-max"    # 截图转写
    model_embed: str = ""              # 配置可用的 embedding 模型后启用

    data_dir: str = "./data"
    db_path: str = "./data/home.db"
    blob_dir: str = "./data/blobs"
    timezone: str = "Asia/Shanghai"

    # 消歧阈值（FR-ID-04）
    disambiguate_auto: float = 0.85
    disambiguate_confirm: float = 0.50
    disambiguate_gap: float = 0.10

    # 置信度分档（FR-MEM-02）
    confidence_auto: float = 0.85
    confidence_confirm: float = 0.60

    # 上下文预算（tokens 粗估：1 token ≈ 1.5 个中文字符的倒数，这里用字符数近似）
    ctx_budget_chars: int = 24000
    dossier_card_chars: int = 900       # ≈300 tokens

    # 账本去重窗口（天）
    ledger_dedupe_days: int = 7

    @property
    def abs_db_path(self) -> Path:
        p = Path(self.db_path)
        return p if p.is_absolute() else ROOT / p

    @property
    def abs_blob_dir(self) -> Path:
        p = Path(self.blob_dir)
        return p if p.is_absolute() else ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()


def tz():
    from zoneinfo import ZoneInfo
    return ZoneInfo(get_settings().timezone)
