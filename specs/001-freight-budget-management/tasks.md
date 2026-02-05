# Tasks: Freight Budget Management

**Input**: Design documents from `/specs/001-freight-budget-management/`  
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are optional and not requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create package skeleton in `freight_budget_management/` (module directories and `__init__.py`)
- [x] T002 [P] Add ReportLab dependency to `requirements.txt`
- [x] T003 [P] Create base app wiring in `app.py` to mount FreightBudgetManagement dashboard entrypoint
- [x] T004 [P] Add local config defaults in `freight_budget_management/storage/db.py` for SQLite path and PDF output directory

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T005 Implement SQLite schema creation and migrations bootstrap in `freight_budget_management/storage/db.py`
- [x] T006 [P] Implement repository interfaces in `freight_budget_management/storage/repositories.py` for quotations, line items, audit events
- [x] T007 [P] Define lifecycle state constants and transition rules in `freight_budget_management/domain/lifecycle.py` (from `specs/freight_budget_management.spec.yaml`)
- [x] T008 [P] Define command definitions and validation mappings in `freight_budget_management/domain/commands.py` (from `specs/freight_budget_management.spec.yaml`)
- [x] T009 [P] Implement validation helpers in `freight_budget_management/domain/validations.py` for line item totals, dates, and version rules
- [x] T010 Implement audit event recording helper in `freight_budget_management/services/audit_service.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Manage freight quotations lifecycle (Priority: P1) 🎯 MVP

**Goal**: Create, transition, and version quotations while enforcing lifecycle rules and immutability.

**Independent Test**: Create a quotation, transition through allowed states, attempt edit after immutability and verify new version creation plus audit events.

### Implementation for User Story 1

- [x] T011 [P] [US1] Implement quotation repository operations in `freight_budget_management/storage/repositories.py` (create, fetch latest, fetch version)
- [x] T012 [P] [US1] Implement line item persistence in `freight_budget_management/storage/repositories.py`
- [x] T013 [US1] Implement quotation command execution in `freight_budget_management/services/quotation_service.py`
- [x] T014 [US1] Enforce lifecycle transitions and immutability/versioning in `freight_budget_management/services/quotation_service.py`
- [x] T015 [US1] Persist audit events for all command attempts in `freight_budget_management/services/audit_service.py`
- [x] T016 [US1] Implement Streamlit UI for create/retrieve/command actions in `freight_budget_management/web/dashboard.py`
- [x] T017 [US1] Wire UI actions to services in `app.py`

**Checkpoint**: User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Generate standardized quotation PDFs (Priority: P2)

**Goal**: Generate standardized PDFs, including “Draft/Unapproved” watermark for non-approved quotations.

**Independent Test**: Generate PDF for approved and non-approved quotations and verify content and watermark behavior.

### Implementation for User Story 2

- [x] T018 [P] [US2] Implement PDF rendering with watermark support in `freight_budget_management/pdf/renderer.py`
- [x] T019 [US2] Add PDF generation workflow in `freight_budget_management/services/quotation_service.py`
- [x] T020 [US2] Add UI action to generate and download PDFs in `freight_budget_management/web/dashboard.py`

**Checkpoint**: User Stories 1 and 2 should both work independently

---

## Phase 5: User Story 3 - Monitor budget and quotation metrics (Priority: P3)

**Goal**: Provide dashboard-ready metrics including counts, financial totals, and conversion rates with date/status filters.

**Independent Test**: Load sample quotations and verify metrics and conversion rate accuracy for a date range.

### Implementation for User Story 3

- [x] T021 [P] [US3] Implement metrics aggregation in `freight_budget_management/services/metrics_service.py`
- [x] T022 [US3] Add metrics view and filters in `freight_budget_management/web/dashboard.py`
- [x] T023 [US3] Add metrics export/download option (CSV) in `freight_budget_management/web/dashboard.py`

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T024 [P] Update usage notes in `README.md` for freight budget workflows
- [x] T025 [P] Validate `specs/001-freight-budget-management/quickstart.md` steps against the implementation
- [x] T026 [P] Add basic error messaging in `freight_budget_management/web/dashboard.py` for failed commands and PDF generation
- [x] T027 [P] Add data export utility in `freight_budget_management/services/metrics_service.py` for dashboard metrics

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can proceed in parallel (if staffed) or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Depends on US1 data and services for quotation retrieval
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Depends on US1 data for metrics

### Within Each User Story

- Repositories before services
- Services before UI wiring
- Story complete before moving to next priority

### Parallel Opportunities

- Setup tasks T002–T004 can run in parallel
- Foundational tasks T006–T009 can run in parallel
- User story tasks marked [P] can run in parallel within each story

---

## Parallel Example: User Story 1

```bash
# Launch repository tasks together:
Task: "Implement quotation repository operations in freight_budget_management/storage/repositories.py"
Task: "Implement line item persistence in freight_budget_management/storage/repositories.py"
```

---

## Parallel Example: User Story 2

```bash
# PDF rendering can be built in parallel with UI wiring:
Task: "Implement PDF rendering with watermark support in freight_budget_management/pdf/renderer.py"
Task: "Add UI action to generate and download PDFs in freight_budget_management/web/dashboard.py"
```

---

## Parallel Example: User Story 3

```bash
# Metrics aggregation can be built in parallel with UI wiring:
Task: "Implement metrics aggregation in freight_budget_management/services/metrics_service.py"
Task: "Add metrics view and filters in freight_budget_management/web/dashboard.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Demo (MVP)
3. Add User Story 2 → Test independently → Demo
4. Add User Story 3 → Test independently → Demo
5. Each story adds value without breaking previous stories
