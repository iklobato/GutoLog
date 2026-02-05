"""Repository interfaces for Freight Budget Management storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Optional

from freight_budget_management.storage.db import get_connection


@dataclass(frozen=True)
class QuotationRecord:
    quotation_id: str
    version: int
    status: str
    created_at: datetime
    updated_at: datetime
    valid_from: str
    valid_to: str
    currency: str
    total_amount: float
    budget_allocation_id: Optional[str]
    customer_name: Optional[str]


@dataclass(frozen=True)
class QuotationLineItemRecord:
    quotation_id: str
    version: int
    description: str
    quantity: float
    unit_price: float
    amount: float


@dataclass(frozen=True)
class AuditEventRecord:
    audit_event_id: str
    quotation_id: Optional[str]
    version: Optional[int]
    command_name: str
    actor: str
    timestamp: datetime
    outcome: str
    reason: Optional[str]
    payload_json: Optional[str]


@dataclass(frozen=True)
class EditLockRecord:
    quotation_id: str
    locked_by: str
    locked_at: datetime
    lock_status: str


class QuotationRepository:
    def __init__(self, connection_factory: Callable = get_connection) -> None:
        self._connection_factory = connection_factory

    def create(self, record: QuotationRecord) -> None:
        with self._connection_factory() as connection:
            connection.execute(
                """
                INSERT INTO quotations (
                    quotation_id, version, status, created_at, updated_at,
                    valid_from, valid_to, currency, total_amount, budget_allocation_id,
                    customer_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.quotation_id,
                    record.version,
                    record.status,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.valid_from,
                    record.valid_to,
                    record.currency,
                    record.total_amount,
                    record.budget_allocation_id,
                    record.customer_name,
                ),
            )

    def get_latest(self, quotation_id: str) -> Optional[QuotationRecord]:
        with self._connection_factory() as connection:
            row = connection.execute(
                """
                SELECT * FROM quotations
                WHERE quotation_id = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (quotation_id,),
            ).fetchone()
        return _row_to_quotation(row) if row else None

    def get_version(self, quotation_id: str, version: int) -> Optional[QuotationRecord]:
        with self._connection_factory() as connection:
            row = connection.execute(
                """
                SELECT * FROM quotations
                WHERE quotation_id = ? AND version = ?
                """,
                (quotation_id, version),
            ).fetchone()
        return _row_to_quotation(row) if row else None

    def list_by_status(self, status: Optional[str] = None) -> Iterable[QuotationRecord]:
        with self._connection_factory() as connection:
            if status:
                rows = connection.execute(
                    "SELECT * FROM quotations WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM quotations ORDER BY created_at DESC",
                ).fetchall()
        return [_row_to_quotation(row) for row in rows]

    def list_filtered(
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
        clauses = []
        params: list = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if customer:
            clauses.append("customer_name LIKE ?")
            params.append(f"%{customer}%")
        if min_amount is not None:
            clauses.append("total_amount >= ?")
            params.append(min_amount)
        if max_amount is not None:
            clauses.append("total_amount <= ?")
            params.append(max_amount)
        if start_date:
            clauses.append("date(updated_at) >= date(?)")
            params.append(start_date)
        if end_date:
            clauses.append("date(updated_at) <= date(?)")
            params.append(end_date)

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connection_factory() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM quotations
                {where_clause}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_row_to_quotation(row) for row in rows]

    def update(self, quotation_id: str, version: int, fields: dict) -> None:
        if not fields:
            return
        set_clause = ", ".join(f"{field} = ?" for field in fields)
        values = list(fields.values()) + [quotation_id, version]
        with self._connection_factory() as connection:
            connection.execute(
                f"UPDATE quotations SET {set_clause} WHERE quotation_id = ? AND version = ?",
                values,
            )


class LineItemRepository:
    def __init__(self, connection_factory: Callable = get_connection) -> None:
        self._connection_factory = connection_factory

    def replace_for_version(self, quotation_id: str, version: int, items: Iterable[QuotationLineItemRecord]) -> None:
        with self._connection_factory() as connection:
            connection.execute(
                "DELETE FROM quotation_line_items WHERE quotation_id = ? AND version = ?",
                (quotation_id, version),
            )
            connection.executemany(
                """
                INSERT INTO quotation_line_items (
                    quotation_id, version, description, quantity, unit_price, amount
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.quotation_id,
                        item.version,
                        item.description,
                        item.quantity,
                        item.unit_price,
                        item.amount,
                    )
                    for item in items
                ],
            )

    def list_for_version(self, quotation_id: str, version: int) -> Iterable[QuotationLineItemRecord]:
        with self._connection_factory() as connection:
            rows = connection.execute(
                """
                SELECT quotation_id, version, description, quantity, unit_price, amount
                FROM quotation_line_items
                WHERE quotation_id = ? AND version = ?
                """,
                (quotation_id, version),
            ).fetchall()
        return [_row_to_line_item(row) for row in rows]


class AuditEventRepository:
    def __init__(self, connection_factory: Callable = get_connection) -> None:
        self._connection_factory = connection_factory

    def create(self, record: AuditEventRecord) -> None:
        with self._connection_factory() as connection:
            connection.execute(
                """
                INSERT INTO audit_events (
                    audit_event_id, quotation_id, version, command_name, actor,
                    timestamp, outcome, reason, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.audit_event_id,
                    record.quotation_id,
                    record.version,
                    record.command_name,
                    record.actor,
                    record.timestamp.isoformat(),
                    record.outcome,
                    record.reason,
                    record.payload_json,
                ),
            )

    def list_for_quotation(self, quotation_id: str) -> Iterable[AuditEventRecord]:
        with self._connection_factory() as connection:
            rows = connection.execute(
                """
                SELECT * FROM audit_events
                WHERE quotation_id = ?
                ORDER BY timestamp DESC
                """,
                (quotation_id,),
            ).fetchall()
        return [_row_to_audit_event(row) for row in rows]

    def list_history(self, quotation_id: str, limit: int = 200) -> Iterable[AuditEventRecord]:
        with self._connection_factory() as connection:
            rows = connection.execute(
                """
                SELECT * FROM audit_events
                WHERE quotation_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (quotation_id, limit),
            ).fetchall()
        return [_row_to_audit_event(row) for row in rows]


class EditLockRepository:
    def __init__(self, connection_factory: Callable = get_connection) -> None:
        self._connection_factory = connection_factory

    def get_lock(self, quotation_id: str) -> Optional[EditLockRecord]:
        with self._connection_factory() as connection:
            row = connection.execute(
                """
                SELECT * FROM edit_locks
                WHERE quotation_id = ? AND lock_status = 'active'
                """,
                (quotation_id,),
            ).fetchone()
        return _row_to_edit_lock(row) if row else None

    def upsert_lock(self, quotation_id: str, locked_by: str, locked_at: datetime, status: str = "active") -> None:
        with self._connection_factory() as connection:
            connection.execute(
                """
                INSERT INTO edit_locks (quotation_id, locked_by, locked_at, lock_status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(quotation_id) DO UPDATE SET
                    locked_by = excluded.locked_by,
                    locked_at = excluded.locked_at,
                    lock_status = excluded.lock_status
                """,
                (quotation_id, locked_by, locked_at.isoformat(), status),
            )

    def release_lock(self, quotation_id: str, locked_by: str) -> None:
        with self._connection_factory() as connection:
            connection.execute(
                """
                UPDATE edit_locks
                SET lock_status = 'released'
                WHERE quotation_id = ? AND locked_by = ?
                """,
                (quotation_id, locked_by),
            )


def _row_to_quotation(row) -> QuotationRecord:
    return QuotationRecord(
        quotation_id=row["quotation_id"],
        version=row["version"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        currency=row["currency"],
        total_amount=row["total_amount"],
        budget_allocation_id=row["budget_allocation_id"],
        customer_name=row["customer_name"],
    )


def _row_to_line_item(row) -> QuotationLineItemRecord:
    return QuotationLineItemRecord(
        quotation_id=row["quotation_id"],
        version=row["version"],
        description=row["description"],
        quantity=row["quantity"],
        unit_price=row["unit_price"],
        amount=row["amount"],
    )


def _row_to_audit_event(row) -> AuditEventRecord:
    return AuditEventRecord(
        audit_event_id=row["audit_event_id"],
        quotation_id=row["quotation_id"],
        version=row["version"],
        command_name=row["command_name"],
        actor=row["actor"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        outcome=row["outcome"],
        reason=row["reason"],
        payload_json=row["payload_json"],
    )


def _row_to_edit_lock(row) -> EditLockRecord:
    return EditLockRecord(
        quotation_id=row["quotation_id"],
        locked_by=row["locked_by"],
        locked_at=datetime.fromisoformat(row["locked_at"]),
        lock_status=row["lock_status"],
    )
