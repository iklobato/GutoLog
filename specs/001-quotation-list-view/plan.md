# Implementation Plan: Quotation List View

**Branch**: `001-quotation-list-view` | **Date**: 2026-02-04 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `/specs/001-quotation-list-view/spec.md`

## Summary

Add a quotation home page with list + filtering, row navigation to the edit screen, and full audit history visibility, while enforcing edit locks and audit tracking. Implement within the existing Streamlit-based quotation system using the current domain, storage, and service layers.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: streamlit, pandas, openpyxl, plotly, numpy  
**Storage**: SQLite (local file) plus file storage for PDFs  
**Testing**: pytest  
**Target Platform**: Local workstation or internal server (macOS/Linux)  
**Project Type**: single  
**Performance Goals**: 95% of list views with filters load in under 2 seconds for 1,000 quotations  
**Constraints**: use only free tools/services; enforce edit lock on open; full audit history visible on edit screen  
**Scale/Scope**: small-to-medium operations (at least 1,000 quotations in list view)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Constitution file contains placeholders only; no enforceable gates found. Gate passes by default. Re-check after Phase 1 in case constitution is updated.
- Post-Phase 1 re-check: no constitution changes detected; gate remains pass.

## Project Structure

### Documentation (this feature)

```text
specs/001-quotation-list-view/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
freight_budget_management/
├── domain/
├── services/
├── storage/
└── web/

app.py
tests/
├── unit/
├── integration/
└── contract/
```

**Structure Decision**: Single Streamlit application using the existing freight budget management module and root `app.py`.
