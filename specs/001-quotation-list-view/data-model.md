# Data Model: Quotation List View

## Entity: Quotation
- **Purpose**: Business record containing status, dates, customer, and amount.
- **Fields**:
  - `quotation_id`
  - `version`
  - `status`
  - `customer_name` or `customer_id`
  - `valid_from`, `valid_to`
  - `total_amount`
  - `updated_at`
- **Relationships**:
  - One-to-many with **Audit Entry**

## Entity: Audit Entry
- **Purpose**: Immutable record of field-level changes.
- **Fields**:
  - `audit_event_id`
  - `quotation_id`
  - `version`
  - `actor`
  - `timestamp`
  - `before_values`
  - `after_values`

## Entity: Edit Lock
- **Purpose**: Prevent concurrent edits by tracking active editor.
- **Fields**:
  - `quotation_id`
  - `locked_by`
  - `locked_at`
  - `lock_status` (active/released)

## Validation Rules
- Only one active lock per quotation.
- Lock released only on explicit save or close action.
