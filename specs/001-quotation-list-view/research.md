# Research Summary: Quotation List View

## Decision: Streamlit-based UI for list and edit flows
- **Rationale**: Existing system already uses Streamlit for quotations; reuse keeps UX consistent.
- **Alternatives considered**: Separate web framework with dedicated frontend.

## Decision: SQLite for quotation listing and audit retrieval
- **Rationale**: Current storage is SQLite; supports listing, filters, and audit reads without new infrastructure.
- **Alternatives considered**: External database service.

## Decision: In-app edit lock tracking
- **Rationale**: Matches single-editor requirement while keeping dependencies minimal.
- **Alternatives considered**: Distributed lock service.
