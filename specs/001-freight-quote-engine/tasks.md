# Tasks: Freight Quote Engine

**Input**: Design documents from `/specs/001-freight-quote-engine/`  
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

- [x] T001 Create quotation engine module in `freight_budget_management/domain/quote_engine.py`
- [x] T002 [P] Add static table loader in `freight_budget_management/domain/quote_tables.py`
- [x] T003 [P] Add validation error types in `freight_budget_management/domain/quote_errors.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Implement table schemas and parsing in `freight_budget_management/domain/quote_tables.py`
- [x] T005 [P] Implement request validation in `freight_budget_management/domain/quote_validations.py`
- [x] T006 [P] Implement rounding helper in `freight_budget_management/domain/quote_rounding.py`
- [x] T007 [P] Implement calculation flow in `freight_budget_management/domain/quote_engine.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Calculate a freight quotation (Priority: P1) 🎯 MVP

**Goal**: Compute a quotation using KM band, vehicle type, trip structure, and surcharges.

**Independent Test**: Use known static tables and verify freight_base, surcharges, and totals.

### Implementation for User Story 1

- [x] T008 [P] [US1] Implement KM band selection in `freight_budget_management/domain/quote_engine.py`
- [x] T009 [P] [US1] Implement base freight + quantity logic in `freight_budget_management/domain/quote_engine.py`
- [x] T010 [US1] Implement surcharge calculations in `freight_budget_management/domain/quote_engine.py`

**Checkpoint**: User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Enforce validation rules (Priority: P2)

**Goal**: Reject invalid requests with explicit errors.

**Independent Test**: Submit invalid payloads and verify validation errors for each rule.

### Implementation for User Story 2

- [x] T011 [P] [US2] Implement insurance limit validation in `freight_budget_management/domain/quote_validations.py`
- [x] T012 [P] [US2] Implement vehicle capacity validation in `freight_budget_management/domain/quote_validations.py`
- [x] T013 [US2] Wire validation failures into engine responses in `freight_budget_management/domain/quote_engine.py`

**Checkpoint**: User Stories 1 and 2 should both work independently

---

## Phase 5: User Story 3 - Provide audit-ready output (Priority: P3)

**Goal**: Return full breakdown and reference prices for audit/negotiation.

**Independent Test**: Verify response includes KM band used, reference prices, and breakdowns.

### Implementation for User Story 3

- [x] T014 [P] [US3] Implement insurance breakdown in `freight_budget_management/domain/quote_engine.py`
- [x] T015 [P] [US3] Implement negotiation + tax calculation in `freight_budget_management/domain/quote_engine.py`
- [x] T016 [US3] Assemble audit-ready response payload in `freight_budget_management/domain/quote_engine.py`

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T017 [P] Add engine usage notes in `README.md`
- [x] T018 [P] Validate `specs/001-freight-quote-engine/quickstart.md` steps against implementation

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
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Depends on base engine and validations
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Depends on base engine and validation outputs

### Within Each User Story

- Validations before computation where applicable
- Core computation before response assembly
- Story complete before moving to next priority

### Parallel Opportunities

- Setup tasks T002–T003 can run in parallel
- Foundational tasks T005–T007 can run in parallel
- User story tasks marked [P] can run in parallel within each story

---

## Parallel Example: User Story 1

```bash
# Build KM selection and base freight in parallel:
Task: "Implement KM band selection in freight_budget_management/domain/quote_engine.py"
Task: "Implement base freight + quantity logic in freight_budget_management/domain/quote_engine.py"
```

---

## Parallel Example: User Story 2

```bash
# Add insurance limit and capacity validations in parallel:
Task: "Implement insurance limit validation in freight_budget_management/domain/quote_validations.py"
Task: "Implement vehicle capacity validation in freight_budget_management/domain/quote_validations.py"
```

---

## Parallel Example: User Story 3

```bash
# Implement insurance breakdown and negotiation/tax in parallel:
Task: "Implement insurance breakdown in freight_budget_management/domain/quote_engine.py"
Task: "Implement negotiation + tax calculation in freight_budget_management/domain/quote_engine.py"
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
