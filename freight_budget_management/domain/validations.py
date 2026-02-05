"""Validation helpers for Freight Budget Management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence


@dataclass(frozen=True)
class LineItemInput:
    description: str
    quantity: float
    unit_price: float
    amount: float


def validate_line_items(items: Iterable[LineItemInput]) -> float:
    """Validate line items and return the total amount."""
    total = 0.0
    for item in items:
        expected_amount = round(item.quantity * item.unit_price, 2)
        if round(item.amount, 2) != expected_amount:
            raise ValueError("Line item amount must equal quantity * unit_price")
        total += item.amount
    return round(total, 2)


def validate_dates(valid_from: date, valid_to: date) -> None:
    if valid_from > valid_to:
        raise ValueError("valid_from must be before or equal to valid_to")


def validate_version_sequence(previous_version: int, new_version: int) -> None:
    if new_version != previous_version + 1:
        raise ValueError("Quotation version must increment by 1")


def ensure_required_fields(payload: dict, required_fields: Sequence[str]) -> None:
    missing = [field for field in required_fields if field not in payload or payload[field] is None]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
