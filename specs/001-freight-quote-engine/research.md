# Research Summary: Freight Quote Engine

## Decision: YAML-backed static tables
- **Rationale**: Supports deterministic configuration and auditability without a database.
- **Alternatives considered**: Hardcoded constants, database storage.

## Decision: Decimal rounding at each step
- **Rationale**: Aligns with business rule for currency precision and avoids drift.
- **Alternatives considered**: Final-only rounding.

## Decision: Domain-first calculation module
- **Rationale**: Keeps calculation logic isolated, unit-testable, and reusable.
- **Alternatives considered**: Inline calculations inside API handlers.
