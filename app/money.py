"""Exact currency conversion shared by extraction and write tools."""
from decimal import Decimal, InvalidOperation


# Match the manual ledger API, and keep cents exact in SQLite and browser JSON.
MAX_AMOUNT_CENTS = 100_000_000_000_000


def amount_to_cents(value) -> int:
    """Accept positive amounts representable in cents; never round or accept NaN."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("金额必须为正数，最多两位小数")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("金额必须为正数，最多两位小数") from exc
    if not amount.is_finite() or not Decimal('0.01') <= amount <= Decimal(MAX_AMOUNT_CENTS) / 100:
        raise ValueError("金额超出范围或不是有效正数")
    rounded = amount.quantize(Decimal('0.01'))
    if amount != rounded:
        raise ValueError("金额最多两位小数，不能自动四舍五入")
    return int(rounded * 100)
