# Tasks: Quotation List View

**Input**: Design documents from `/specs/001-quotation-list-view/`  
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

- [x] T001 Verify existing quotation data access in `freight_budget_management/storage/repositories.py`
- [x] T002 [P] Add list and filter query helpers in `freight_budget_management/storage/repositories.py`
- [x] T003 [P] Add edit lock storage helpers in `freight_budget_management/storage/repositories.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Add edit lock table schema to `freight_budget_management/storage/db.py`
- [x] T005 [P] Add audit history query helper in `freight_budget_management/storage/repositories.py`
- [x] T006 [P] Add quotation list/filter service in `freight_budget_management/services/quotation_service.py`
- [x] T007 [P] Add edit lock service in `freight_budget_management/services/quotation_service.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - View and filter all quotations (Priority: P1) 🎯 MVP

**Goal**: Show the latest 100 quotations with filters in a list view.

**Independent Test**: Load sample quotations and verify list rendering plus filter behavior.

### Implementation for User Story 1

- [x] T008 [P] [US1] Add list view UI and filters in `freight_budget_management/web/dashboard.py`
- [x] T009 [US1] Wire list/filter UI to service in `freight_budget_management/web/dashboard.py`

**Checkpoint**: User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Open a quotation for editing (Priority: P2)

**Goal**: Navigate from list view to edit screen and enforce edit locks.

**Independent Test**: Open a quotation, confirm details load, then try to open the same quotation as a second user and confirm lock behavior.

### Implementation for User Story 2

- [x] T010 [P] [US2] Add row click navigation to edit screen in `freight_budget_management/web/dashboard.py`
- [x] T011 [US2] Implement lock acquisition/release in `freight_budget_management/services/quotation_service.py`
- [x] T012 [US2] Show lock owner details on edit screen in `freight_budget_management/web/dashboard.py`

**Checkpoint**: User Stories 1 and 2 should both work independently

---

## Phase 5: User Story 3 - Track edits with audit history (Priority: P3)

**Goal**: Allow edits on the edit screen and show full audit history.

**Independent Test**: Edit a quotation and confirm audit history shows before/after values and actor/timestamp.

### Implementation for User Story 3

- [x] T013 [P] [US3] Add edit form and save flow in `freight_budget_management/web/dashboard.py`
- [x] T014 [US3] Persist edits and audit entries in `freight_budget_management/services/quotation_service.py`
- [x] T015 [US3] Display full audit history on edit screen in `freight_budget_management/web/dashboard.py`

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T016 [P] Update usage notes in `README.md` for quotation list and edit flow
- [x] T017 [P] Validate `specs/001-quotation-list-view/quickstart.md` steps against implementation
- [x] T018 [P] Add basic error messaging for lock conflicts in `freight_budget_management/web/dashboard.py`

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
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Depends on US1 list view and lock services
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Depends on US2 edit screen and audit storage

### Within Each User Story

- Services before UI wiring
- Story complete before moving to next priority

### Parallel Opportunities

- Setup tasks T002–T003 can run in parallel
- Foundational tasks T005–T007 can run in parallel
- User story tasks marked [P] can run in parallel within each story

---

## Parallel Example: User Story 1

```bash
# Build list view UI and filters in parallel:
Task: "Add list view UI and filters in freight_budget_management/web/dashboard.py"
Task: "Wire list/filter UI to service in freight_budget_management/web/dashboard.py"
```

---

## Parallel Example: User Story 2

```bash
# Implement UI navigation and lock service in parallel:
Task: "Add row click navigation to edit screen in freight_budget_management/web/dashboard.py"
Task: "Implement lock acquisition/release in freight_budget_management/services/quotation_service.py"
```

---

## Parallel Example: User Story 3

```bash
# Build edit form and audit view in parallel:
Task: "Add edit form and save flow in freight_budget_management/web/dashboard.py"
Task: "Display full audit history on edit screen in freight_budget_management/web/dashboard.py"
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
