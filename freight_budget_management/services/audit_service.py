"""Audit event recording for Freight Budget Management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from freight_budget_management.storage.repositories import AuditEventRecord, AuditEventRepository


@dataclass(frozen=True)
class AuditResult:
    audit_event_id: str


class AuditService:
    def __init__(self, repository: AuditEventRepository) -> None:
        self._repository = repository

    def record_event(
        self,
        *,
        quotation_id: Optional[str],
        version: Optional[int],
        command_name: str,
        actor: str,
        outcome: str,
        reason: Optional[str] = None,
        payload_json: Optional[str] = None,
    ) -> AuditResult:
        audit_event_id = str(uuid4())
        record = AuditEventRecord(
            audit_event_id=audit_event_id,
            quotation_id=quotation_id,
            version=version,
            command_name=command_name,
            actor=actor,
            timestamp=datetime.now(timezone.utc),
            outcome=outcome,
            reason=reason,
            payload_json=payload_json,
        )
        self._repository.create(record)
        return AuditResult(audit_event_id=audit_event_id)
