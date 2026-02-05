"""Metrics aggregation for Freight Budget Management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Optional

from freight_budget_management.domain import lifecycle
from freight_budget_management.storage.repositories import QuotationRepository


@dataclass(frozen=True)
class MetricsResult:
    status_counts: Dict[str, int]
    total_amounts: Dict[str, float]
    conversion_rate: float


class MetricsService:
    def __init__(self, quotation_repository: QuotationRepository, spec: Optional[dict] = None) -> None:
        self._quotation_repository = quotation_repository
        self._spec = spec or lifecycle.load_spec()

    def get_metrics(
        self,
        *,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        status: Optional[str] = None,
    ) -> MetricsResult:
        records = list(self._quotation_repository.list_by_status(status))
        filtered = [
            record
            for record in records
            if _in_date_range(record.created_at, start_date, end_date)
        ]

        status_counts: Dict[str, int] = {}
        total_amounts: Dict[str, float] = {}
        for record in filtered:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1
            total_amounts[record.status] = total_amounts.get(record.status, 0.0) + record.total_amount

        conversion_rate = _compute_conversion_rate(self._spec, status_counts)
        return MetricsResult(status_counts=status_counts, total_amounts=total_amounts, conversion_rate=conversion_rate)

    @staticmethod
    def export_metrics_csv(result: MetricsResult) -> str:
        lines = ["metric,value"]
        lines.append(f"conversion_rate,{result.conversion_rate}")
        for status, count in result.status_counts.items():
            lines.append(f"count_{status},{count}")
        for status, total in result.total_amounts.items():
            lines.append(f"total_{status},{total}")
        return "\n".join(lines) + "\n"


def _in_date_range(value: datetime, start_date: Optional[date], end_date: Optional[date]) -> bool:
    if start_date and value.date() < start_date:
        return False
    if end_date and value.date() > end_date:
        return False
    return True


def _compute_conversion_rate(spec: dict, status_counts: Dict[str, int]) -> float:
    metrics_spec = spec.get("metrics", {})
    from_state = metrics_spec.get("conversion_from") or lifecycle.get_initial_state(spec)
    to_state = metrics_spec.get("conversion_to") or "approved"
    from_count = status_counts.get(from_state, 0)
    to_count = status_counts.get(to_state, 0)
    if from_count == 0:
        return 0.0
    return round(to_count / from_count, 4)
