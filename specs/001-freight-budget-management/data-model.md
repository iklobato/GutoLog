# Data Model: Freight Budget Management

## Entity: Quotation
- **Purpose**: Freight pricing proposal with lifecycle state and versioning.
- **Fields**:
  - `quotation_id` (stable identifier)
  - `version` (integer revision)
  - `status` (lifecycle state from specification)
  - `created_at`, `updated_at`
  - `valid_from`, `valid_to`
  - `currency`
  - `total_amount`
  - `budget_allocation_id` (optional link)
- **Relationships**:
  - One-to-many with **Quotation Line Item**
  - Many-to-one with **Budget Allocation** (optional)
  - One-to-many with **Audit Event**
- **Validation Rules**:
  - `version` increments only when a new version is created.
  - `status` transitions must follow the specification.
  - `valid_from` <= `valid_to`.

## Entity: Quotation Line Item
- **Purpose**: Individual charge components that sum to a quotation total.
- **Fields**:
  - `line_item_id`
  - `quotation_id`, `version`
  - `description`
  - `quantity`
  - `unit_price`
  - `amount`
- **Relationships**:
  - Belongs to **Quotation**.
- **Validation Rules**:
  - `amount` equals `quantity * unit_price`.
  - Sum of line item amounts equals `Quotation.total_amount`.

## Entity: Budget Allocation
- **Purpose**: Planned or available budget tied to quotations.
- **Fields**:
  - `budget_allocation_id`
  - `name`
  - `amount_available`
  - `currency`
  - `valid_from`, `valid_to`
- **Relationships**:
  - One-to-many with **Quotation**.

## Entity: Audit Event
- **Purpose**: Immutable record of every command attempt and outcome.
- **Fields**:
  - `audit_event_id`
  - `quotation_id`, `version` (if applicable)
  - `command_name`
  - `actor`
  - `timestamp`
  - `outcome` (success/failure)
  - `reason` (failure reason, if any)

## Entity: Metric Snapshot (derived)
- **Purpose**: Aggregated measures for dashboard reporting.
- **Fields**:
  - `period_start`, `period_end`
  - `status_counts`
  - `total_amounts`
  - `conversion_rate`

## Lifecycle / State Transitions
- States and transitions are defined by `specs/freight_budget_management.spec.yaml`.
- Immutable state requires versioned updates: create a new version; prior version remains unchanged.
