"""Quotation command execution and lifecycle enforcement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from freight_budget_management.domain import commands as command_rules
from freight_budget_management.domain import lifecycle
from freight_budget_management.domain.validations import (
    LineItemInput,
    ensure_required_fields,
    validate_dates,
    validate_line_items,
    validate_version_sequence,
)
from freight_budget_management.pdf.renderer import render_quotation_pdf
from freight_budget_management.services.audit_service import AuditService
from freight_budget_management.storage.db import get_pdf_output_dir
from freight_budget_management.storage.repositories import (
    AuditEventRepository,
    EditLockRecord,
    EditLockRepository,
    LineItemRepository,
    QuotationLineItemRecord,
    QuotationRecord,
    QuotationRepository,
)


@dataclass(frozen=True)
class CommandResult:
    success: bool
    quotation: Optional[QuotationRecord]
    audit_event_id: str
    message: str


@dataclass(frozen=True)
class LockResult:
    success: bool
    lock: Optional[EditLockRecord]
    message: str


class QuotationService:
    def __init__(
        self,
        quotation_repository: QuotationRepository,
        line_item_repository: LineItemRepository,
        audit_service: AuditService,
        audit_repository: Optional[AuditEventRepository] = None,
        edit_lock_repository: Optional[EditLockRepository] = None,
        spec: Optional[dict] = None,
    ) -> None:
        self._quotation_repository = quotation_repository
        self._line_item_repository = line_item_repository
        self._audit_service = audit_service
        self._audit_repository = audit_repository or AuditEventRepository()
        self._edit_lock_repository = edit_lock_repository or EditLockRepository()
        self._spec = spec or lifecycle.load_spec()

    def create_quotation(self, payload: Dict[str, Any], actor: str) -> CommandResult:
        command_name = payload.get("command_name", "create_quotation")
        try:
            required_fields = ["currency", "valid_from", "valid_to", "line_items"]
            ensure_required_fields(payload, required_fields)
            line_items_input = _parse_line_items(payload["line_items"])
            total_amount = validate_line_items(line_items_input)
            validate_dates(payload["valid_from"], payload["valid_to"])

            now = datetime.now(timezone.utc)
            quotation_id = payload.get("quotation_id") or _generate_quotation_id()
            record = QuotationRecord(
                quotation_id=quotation_id,
                version=1,
                status=lifecycle.get_initial_state(self._spec),
                created_at=now,
                updated_at=now,
                valid_from=str(payload["valid_from"]),
                valid_to=str(payload["valid_to"]),
                currency=payload["currency"],
                total_amount=total_amount,
                budget_allocation_id=payload.get("budget_allocation_id"),
                customer_name=payload.get("customer_name"),
            )
            self._quotation_repository.create(record)
            self._line_item_repository.replace_for_version(
                quotation_id,
                1,
                _line_item_records(quotation_id, 1, line_items_input),
            )
            audit_result = self._audit_service.record_event(
                quotation_id=quotation_id,
                version=1,
                command_name=command_name,
                actor=actor,
                outcome="success",
                payload_json=json.dumps(payload, default=str),
            )
            return CommandResult(True, record, audit_result.audit_event_id, "Quotation created")
        except Exception as exc:  # noqa: BLE001 - ensure audit logging for failures
            audit_result = self._audit_service.record_event(
                quotation_id=payload.get("quotation_id"),
                version=payload.get("version"),
                command_name=command_name,
                actor=actor,
                outcome="failure",
                reason=str(exc),
                payload_json=json.dumps(payload, default=str),
            )
            return CommandResult(False, None, audit_result.audit_event_id, str(exc))

    def execute_command(
        self,
        quotation_id: str,
        command_name: str,
        payload: Dict[str, Any],
        actor: str,
    ) -> CommandResult:
        try:
            definition = command_rules.get_command_definition(command_name, self._spec)
            if definition.required_fields:
                ensure_required_fields(payload, list(definition.required_fields))

            current = self._quotation_repository.get_latest(quotation_id)
            if not current:
                raise ValueError("Quotation not found")

            if definition.allowed_states and current.status not in definition.allowed_states:
                raise ValueError("Command not allowed in current state")

            immutable = lifecycle.is_immutable_state(self._spec, current.status)
            create_new_version = bool(definition.raw.get("creates_new_version") or definition.raw.get("versioned"))
            if immutable and not create_new_version:
                raise ValueError("Quotation is immutable; create a new version to modify")

            line_items_input = _parse_line_items(payload.get("line_items", []))
            if line_items_input:
                total_amount = validate_line_items(line_items_input)
            else:
                total_amount = current.total_amount

            next_status = (
                definition.raw.get("next_state")
                or definition.raw.get("to_state")
                or definition.raw.get("status")
                or current.status
            )

            now = datetime.now(timezone.utc)
            if create_new_version:
                new_version = current.version + 1
                validate_version_sequence(current.version, new_version)
                record = QuotationRecord(
                    quotation_id=current.quotation_id,
                    version=new_version,
                    status=next_status,
                    created_at=current.created_at,
                    updated_at=now,
                    valid_from=str(payload.get("valid_from", current.valid_from)),
                    valid_to=str(payload.get("valid_to", current.valid_to)),
                    currency=payload.get("currency", current.currency),
                    total_amount=total_amount,
                    budget_allocation_id=payload.get("budget_allocation_id", current.budget_allocation_id),
                )
                self._quotation_repository.create(record)
                if line_items_input:
                    self._line_item_repository.replace_for_version(
                        current.quotation_id,
                        new_version,
                        _line_item_records(current.quotation_id, new_version, line_items_input),
                    )
                audit_result = self._audit_service.record_event(
                    quotation_id=current.quotation_id,
                    version=new_version,
                    command_name=command_name,
                    actor=actor,
                    outcome="success",
                    payload_json=json.dumps(payload, default=str),
                )
                return CommandResult(True, record, audit_result.audit_event_id, "Command executed")

            updated_fields = {
                "status": next_status,
                "updated_at": now.isoformat(),
                "valid_from": str(payload.get("valid_from", current.valid_from)),
                "valid_to": str(payload.get("valid_to", current.valid_to)),
                "currency": payload.get("currency", current.currency),
                "total_amount": total_amount,
                "budget_allocation_id": payload.get("budget_allocation_id", current.budget_allocation_id),
            }
            self._quotation_repository.update(current.quotation_id, current.version, updated_fields)
            if line_items_input:
                self._line_item_repository.replace_for_version(
                    current.quotation_id,
                    current.version,
                    _line_item_records(current.quotation_id, current.version, line_items_input),
                )
            refreshed = self._quotation_repository.get_version(current.quotation_id, current.version)
            audit_result = self._audit_service.record_event(
                quotation_id=current.quotation_id,
                version=current.version,
                command_name=command_name,
                actor=actor,
                outcome="success",
                payload_json=json.dumps(payload, default=str),
            )
            return CommandResult(True, refreshed, audit_result.audit_event_id, "Command executed")
        except Exception as exc:  # noqa: BLE001 - ensure audit logging for failures
            audit_result = self._audit_service.record_event(
                quotation_id=quotation_id,
                version=payload.get("version"),
                command_name=command_name,
                actor=actor,
                outcome="failure",
                reason=str(exc),
                payload_json=json.dumps(payload, default=str),
            )
            return CommandResult(False, None, audit_result.audit_event_id, str(exc))

    def generate_pdf(self, quotation_id: str, actor: str) -> CommandResult:
        command_name = "generate_pdf"
        try:
            quotation = self._quotation_repository.get_latest(quotation_id)
            if not quotation:
                raise ValueError("Quotation not found")
            line_items = list(self._line_item_repository.list_for_version(quotation_id, quotation.version))
            watermark_text = _pdf_watermark(self._spec, quotation.status)
            output_dir = get_pdf_output_dir()
            output_path = output_dir / f"{quotation_id}_v{quotation.version}.pdf"
            render_quotation_pdf(quotation, line_items, output_path, watermark_text=watermark_text)
            audit_result = self._audit_service.record_event(
                quotation_id=quotation.quotation_id,
                version=quotation.version,
                command_name=command_name,
                actor=actor,
                outcome="success",
            )
            return CommandResult(True, quotation, audit_result.audit_event_id, str(output_path))
        except Exception as exc:  # noqa: BLE001
            audit_result = self._audit_service.record_event(
                quotation_id=quotation_id,
                version=None,
                command_name=command_name,
                actor=actor,
                outcome="failure",
                reason=str(exc),
            )
            return CommandResult(False, None, audit_result.audit_event_id, str(exc))

    def list_quotations(
        self,
        *,
        status: Optional[str] = None,
        customer: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> Iterable[QuotationRecord]:
        return self._quotation_repository.list_filtered(
            status=status,
            customer=customer,
            min_amount=min_amount,
            max_amount=max_amount,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

    def get_audit_history(self, quotation_id: str) -> Iterable[dict]:
        events = self._audit_repository.list_history(quotation_id)
        return [
            {
                "audit_event_id": event.audit_event_id,
                "actor": event.actor,
                "timestamp": event.timestamp,
                "payload_json": event.payload_json,
                "command_name": event.command_name,
                "outcome": event.outcome,
            }
            for event in events
        ]

    def acquire_edit_lock(self, quotation_id: str, actor: str) -> LockResult:
        existing = self._edit_lock_repository.get_lock(quotation_id)
        if existing and existing.locked_by != actor:
            return LockResult(False, existing, f"Locked by {existing.locked_by}")
        now = datetime.now(timezone.utc)
        self._edit_lock_repository.upsert_lock(quotation_id, actor, now, status="active")
        return LockResult(True, self._edit_lock_repository.get_lock(quotation_id), "Lock acquired")

    def release_edit_lock(self, quotation_id: str, actor: str) -> LockResult:
        existing = self._edit_lock_repository.get_lock(quotation_id)
        if not existing:
            return LockResult(True, None, "No active lock")
        if existing.locked_by != actor:
            return LockResult(False, existing, f"Locked by {existing.locked_by}")
        self._edit_lock_repository.release_lock(quotation_id, actor)
        return LockResult(True, None, "Lock released")

    def update_quotation_fields(self, quotation_id: str, fields: Dict[str, Any], actor: str) -> CommandResult:
        command_name = "edit_quotation"
        try:
            current = self._quotation_repository.get_latest(quotation_id)
            if not current:
                raise ValueError("Quotation not found")
            if lifecycle.is_immutable_state(self._spec, current.status):
                raise ValueError("Quotation is immutable")

            lock = self._edit_lock_repository.get_lock(quotation_id)
            if not lock or lock.locked_by != actor:
                raise ValueError("Active lock required to edit")

            allowed_fields = _editable_fields(self._spec)
            filtered = {key: value for key, value in fields.items() if key in allowed_fields}
            if not filtered:
                raise ValueError("No editable fields provided")

            before_values = {key: getattr(current, key) for key in filtered.keys()}
            updated_fields = {**filtered, "updated_at": datetime.now(timezone.utc).isoformat()}
            self._quotation_repository.update(quotation_id, current.version, updated_fields)
            refreshed = self._quotation_repository.get_version(quotation_id, current.version)
            after_values = {key: getattr(refreshed, key) for key in filtered.keys()} if refreshed else filtered

            audit_payload = {
                "before_values": before_values,
                "after_values": after_values,
            }
            audit_result = self._audit_service.record_event(
                quotation_id=quotation_id,
                version=current.version,
                command_name=command_name,
                actor=actor,
                outcome="success",
                payload_json=json.dumps(audit_payload, default=str),
            )
            return CommandResult(True, refreshed, audit_result.audit_event_id, "Quotation updated")
        except Exception as exc:  # noqa: BLE001
            audit_result = self._audit_service.record_event(
                quotation_id=quotation_id,
                version=None,
                command_name=command_name,
                actor=actor,
                outcome="failure",
                reason=str(exc),
            )
            return CommandResult(False, None, audit_result.audit_event_id, str(exc))


def _parse_line_items(raw_items: Iterable[Dict[str, Any]]) -> Iterable[LineItemInput]:
    return [
        LineItemInput(
            description=item["description"],
            quantity=float(item["quantity"]),
            unit_price=float(item["unit_price"]),
            amount=float(item.get("amount", float(item["quantity"]) * float(item["unit_price"]))),
        )
        for item in raw_items
    ]


def _line_item_records(
    quotation_id: str, version: int, items: Iterable[LineItemInput]
) -> Iterable[QuotationLineItemRecord]:
    return [
        QuotationLineItemRecord(
            quotation_id=quotation_id,
            version=version,
            description=item.description,
            quantity=item.quantity,
            unit_price=item.unit_price,
            amount=item.amount,
        )
        for item in items
    ]


def _generate_quotation_id() -> str:
    return datetime.now(timezone.utc).strftime("QT%Y%m%d%H%M%S")


def _pdf_watermark(spec: dict, status: str) -> Optional[str]:
    approved_states = set(
        spec.get("pdf", {}).get("approved_states")
        or spec.get("lifecycle", {}).get("approved_states")
        or []
    )
    if approved_states and status in approved_states:
        return None
    if not approved_states and status.lower() == "approved":
        return None
    return "Draft/Unapproved"


def _editable_fields(spec: dict) -> set:
    fields = spec.get("editable_fields")
    if fields:
        return set(fields)
    return {
        "customer_name",
        "valid_from",
        "valid_to",
        "currency",
        "total_amount",
        "budget_allocation_id",
    }
