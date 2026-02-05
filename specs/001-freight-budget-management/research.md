# Research Summary: Freight Budget Management

## Decision: Python 3.11 runtime
- **Rationale**: Existing project configuration specifies Python >=3.11 and aligns with current tooling.
- **Alternatives considered**: Python 3.10.

## Decision: Streamlit UI layer with domain module
- **Rationale**: Project already uses Streamlit for user-facing workflows; fits interactive dashboard needs.
- **Alternatives considered**: Separate web framework with REST UI.

## Decision: SQLite for persistent storage
- **Rationale**: Free, file-based persistence suitable for small-to-medium deployments and local operations.
- **Alternatives considered**: PostgreSQL, filesystem-only JSON storage.

## Decision: ReportLab for PDF generation
- **Rationale**: Free, Python-native library for precise PDF generation and watermarking requirements.
- **Alternatives considered**: WeasyPrint, wkhtmltopdf.

## Decision: pytest for testing
- **Rationale**: Standard Python testing framework with broad ecosystem support.
- **Alternatives considered**: unittest, nose.
