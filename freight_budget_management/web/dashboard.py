"""Streamlit dashboard for Freight Budget Management."""

from __future__ import annotations

import json

import streamlit as st

from freight_budget_management.services.audit_service import AuditService
from freight_budget_management.domain.lifecycle import load_spec
from freight_budget_management.services.metrics_service import MetricsService
from freight_budget_management.services.quotation_service import QuotationService
from freight_budget_management.storage.db import init_db
from freight_budget_management.storage.repositories import (
    AuditEventRepository,
    EditLockRepository,
    LineItemRepository,
    QuotationRepository,
)


def render() -> None:
    """Render the Freight Budget Management dashboard."""
    st.title("Freight Budget Management")
    init_db()

    try:
        spec = load_spec()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "Add the specification file at `specs/freight_budget_management.spec.yaml` "
            "or set `FREIGHT_BUDGET_SPEC_PATH` to a valid path."
        )
        return

    quotation_repository = QuotationRepository()
    line_item_repository = LineItemRepository()
    audit_repository = AuditEventRepository()
    lock_repository = EditLockRepository()
    audit_service = AuditService(audit_repository)
    quotation_service = QuotationService(
        quotation_repository,
        line_item_repository,
        audit_service,
        audit_repository=audit_repository,
        edit_lock_repository=lock_repository,
        spec=spec,
    )
    metrics_service = MetricsService(quotation_repository, spec=spec)

    tab_list, tab_create, tab_commands, tab_lookup, tab_metrics = st.tabs(
        ["Quotation List", "Create Quotation", "Execute Command", "Retrieve Quotation", "Metrics"]
    )

    with tab_list:
        st.subheader("Quotation List")
        actor = st.text_input("Current User", value="system", key="list_actor")
        status_filter = st.text_input("Status", value="", key="list_status")
        customer_filter = st.text_input("Customer", value="", key="list_customer")
        use_amount_filter = st.checkbox("Filter by Amount", value=False, key="list_amount_enabled")
        min_amount = st.number_input("Min Amount", value=0.0, step=1.0, key="list_min_amount")
        max_amount = st.number_input("Max Amount", value=0.0, step=1.0, key="list_max_amount")
        use_date_filter = st.checkbox("Filter by Date Range", value=False, key="list_date_enabled")
        start_date = st.date_input("Start Date", key="list_start_date")
        end_date = st.date_input("End Date", key="list_end_date")
        if st.button("Refresh List", key="list_refresh"):
            st.session_state.quotation_list = list(
                quotation_service.list_quotations(
                    status=status_filter or None,
                    customer=customer_filter or None,
                    min_amount=min_amount if use_amount_filter else None,
                    max_amount=max_amount if use_amount_filter else None,
                    start_date=str(start_date) if use_date_filter else None,
                    end_date=str(end_date) if use_date_filter else None,
                    limit=100,
                )
            )

        quotations = st.session_state.get("quotation_list", [])
        if quotations:
            list_rows = [
                {
                    "quotation_id": q.quotation_id,
                    "version": q.version,
                    "status": q.status,
                    "customer": q.customer_name or "",
                    "total_amount": q.total_amount,
                    "updated_at": q.updated_at.isoformat(),
                }
                for q in quotations
            ]
            list_df = st.dataframe(
                list_rows,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun",
                key="quotation_list_table",
            )
            selection = st.session_state.get("quotation_list_table", {}).get("selection", {})
            selected_rows = selection.get("rows", []) if isinstance(selection, dict) else []
            selected_id = list_rows[selected_rows[0]]["quotation_id"] if selected_rows else None
            if not selected_id:
                selected_id = st.selectbox(
                    "Select quotation to edit",
                    options=[row["quotation_id"] for row in list_rows],
                    key="quotation_select_fallback",
                )

            if selected_id:
                lock_result = quotation_service.acquire_edit_lock(selected_id, actor)
                if not lock_result.success:
                    st.error(lock_result.message)
                else:
                    quotation = quotation_repository.get_latest(selected_id)
                    if quotation:
                        st.markdown("### Edit Quotation")
                        st.write(f"Locked by: {lock_result.lock.locked_by if lock_result.lock else actor}")
                        customer_name = st.text_input(
                            "Customer",
                            value=quotation.customer_name or "",
                            key="edit_customer_name",
                        )
                        valid_from = st.text_input("Valid From", value=quotation.valid_from, key="edit_valid_from")
                        valid_to = st.text_input("Valid To", value=quotation.valid_to, key="edit_valid_to")
                        currency = st.text_input("Currency", value=quotation.currency, key="edit_currency")
                        total_amount = st.number_input(
                            "Total Amount", value=float(quotation.total_amount), key="edit_total_amount"
                        )
                        budget_allocation_id = st.text_input(
                            "Budget Allocation ID",
                            value=quotation.budget_allocation_id or "",
                            key="edit_budget_allocation",
                        )
                        if st.button("Save Changes", key="save_changes"):
                            result = quotation_service.update_quotation_fields(
                                quotation.quotation_id,
                                {
                                    "customer_name": customer_name,
                                    "valid_from": valid_from,
                                    "valid_to": valid_to,
                                    "currency": currency,
                                    "total_amount": total_amount,
                                    "budget_allocation_id": budget_allocation_id or None,
                                },
                                actor=actor,
                            )
                            if result.success:
                                quotation_service.release_edit_lock(quotation.quotation_id, actor)
                                st.success("Changes saved and lock released")
                            else:
                                st.error(result.message)

                        if st.button("Close Edit", key="close_edit"):
                            close_result = quotation_service.release_edit_lock(quotation.quotation_id, actor)
                            if close_result.success:
                                st.info("Lock released")
                            else:
                                st.error(close_result.message)

                        st.markdown("### Audit History")
                        history = quotation_service.get_audit_history(quotation.quotation_id)
                        audit_rows = []
                        for entry in history:
                            payload = {}
                            if entry.get("payload_json"):
                                try:
                                    payload = json.loads(entry["payload_json"])
                                except json.JSONDecodeError:
                                    payload = {}
                            audit_rows.append(
                                {
                                    "timestamp": entry["timestamp"].isoformat(),
                                    "actor": entry["actor"],
                                    "command": entry["command_name"],
                                    "before": payload.get("before_values"),
                                    "after": payload.get("after_values"),
                                }
                            )
                        st.dataframe(audit_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No quotations available. Use filters and refresh.")

    with tab_create:
        st.subheader("Create Quotation")
        currency = st.text_input("Currency", value="USD")
        customer_name = st.text_input("Customer", value="")
        valid_from = st.date_input("Valid From")
        valid_to = st.date_input("Valid To")
        budget_allocation_id = st.text_input("Budget Allocation ID (optional)", value="")
        line_items_payload = st.text_area(
            "Line Items (JSON list)",
            value='[{"description": "Base Freight", "quantity": 1, "unit_price": 1000.0, "amount": 1000.0}]',
            height=140,
        )
        actor = st.text_input("Actor", value="system")
        if st.button("Create Quotation"):
            try:
                line_items = json.loads(line_items_payload)
                result = quotation_service.create_quotation(
                    {
                        "customer_name": customer_name or None,
                        "currency": currency,
                        "valid_from": valid_from,
                        "valid_to": valid_to,
                        "budget_allocation_id": budget_allocation_id or None,
                        "line_items": line_items,
                    },
                    actor=actor,
                )
                if result.success and result.quotation:
                    st.success(f"Created quotation {result.quotation.quotation_id} v{result.quotation.version}")
                else:
                    st.error(result.message)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to create quotation: {exc}")

    with tab_commands:
        st.subheader("Execute Command")
        quotation_id = st.text_input("Quotation ID")
        command_name = st.text_input("Command Name")
        payload_input = st.text_area("Command Payload (JSON)", value="{}", height=120)
        actor = st.text_input("Actor for Command", value="system")
        if st.button("Execute Command"):
            try:
                payload = json.loads(payload_input)
                result = quotation_service.execute_command(
                    quotation_id=quotation_id,
                    command_name=command_name,
                    payload=payload,
                    actor=actor,
                )
                if result.success and result.quotation:
                    st.success(f"Command applied to {result.quotation.quotation_id} v{result.quotation.version}")
                else:
                    st.error(result.message)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to execute command: {exc}")

    with tab_lookup:
        st.subheader("Retrieve Quotation")
        lookup_id = st.text_input("Lookup Quotation ID")
        version = st.number_input("Version (optional)", min_value=0, value=0, step=1)
        if st.button("Load Quotation"):
            if version > 0:
                quotation = quotation_repository.get_version(lookup_id, int(version))
            else:
                quotation = quotation_repository.get_latest(lookup_id)
            if not quotation:
                st.warning("Quotation not found")
            else:
                st.json(quotation.__dict__)
                items = line_item_repository.list_for_version(quotation.quotation_id, quotation.version)
                if items:
                    st.subheader("Line Items")
                    st.json([item.__dict__ for item in items])
                if st.button("Generate PDF"):
                    result = quotation_service.generate_pdf(quotation.quotation_id, actor="system")
                    if result.success:
                        st.success(f"PDF generated at: {result.message}")
                    else:
                        st.error(result.message)

    with tab_metrics:
        st.subheader("Dashboard Metrics")
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        status_filter = st.text_input("Status Filter (optional)", value="")
        if st.button("Refresh Metrics"):
            st.session_state.metrics_result = metrics_service.get_metrics(
                start_date=start_date,
                end_date=end_date,
                status=status_filter or None,
            )

        if "metrics_result" in st.session_state:
            result = st.session_state.metrics_result
            st.metric("Conversion Rate", f"{result.conversion_rate:.2%}")
            st.subheader("Status Counts")
            st.json(result.status_counts)
            st.subheader("Total Amounts")
            st.json(result.total_amounts)
            csv_data = metrics_service.export_metrics_csv(result)
            st.download_button(
                "Download Metrics CSV",
                data=csv_data,
                file_name="freight_budget_metrics.csv",
                mime="text/csv",
            )
