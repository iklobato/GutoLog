"""Rounding helpers for quotation calculations."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def to_decimal(value: float | int | str) -> Decimal:
    return Decimal(str(value))


def round_currency(value: Decimal | float | int | str) -> Decimal:
    return to_decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
