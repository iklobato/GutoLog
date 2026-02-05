# Feature Specification: Freight Quote Engine

**Feature Branch**: `001-freight-quote-engine`  
**Created**: 2026-02-04  
**Status**: Draft  
**Input**: User description: "You are a senior backend engineer and business analyst. Your task is to IMPLEMENT a freight quotation engine that strictly follows the rules, tables, and dependencies described below. Do NOT simplify, guess, or invent values. Every calculation must be explainable and traceable. [full prompt omitted for brevity]"

## Clarifications

### Session 2026-02-04

- Q: How should cargo weight be provided for vehicle capacity validation? → A: Require cargo_weight in the request.
- Q: How should cargo_value_nf above insurance_limit be handled? → A: Reject the quotation with a validation error.
- Q: How should monetary values be rounded? → A: Round to 2 decimals at each step.
- Q: How should toll be applied with multiple vehicles? → A: Apply toll per vehicle.
- Q: How should diárias be applied with multiple vehicles? → A: Apply diárias per vehicle.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Calculate a freight quotation (Priority: P1)

Operations users submit a quotation request and receive a full price breakdown that follows the mandatory calculation order.

**Why this priority**: The core business value is accurate, rule-driven quotation calculation.

**Independent Test**: Can be fully tested by submitting a complete request with known reference tables and verifying every output component.

**Acceptance Scenarios**:

1. **Given** a complete request and reference tables, **When** the user requests a quotation, **Then** the system returns freight base, surcharges, insurance, negotiation adjustment, tax, and final total.
2. **Given** a request with has_return enabled, **When** the quotation is calculated, **Then** the system uses the return price from the KM table and applies quantity.
3. **Given** a request with cargo_weight, **When** the quotation is calculated, **Then** the system validates vehicle capacity against cargo_weight.

---

### User Story 2 - Enforce validation rules (Priority: P2)

Users receive explicit validation errors when inputs violate mandatory rules.

**Why this priority**: Invalid requests must never produce misleading prices.

**Independent Test**: Can be fully tested by submitting invalid inputs and verifying specific error messages.

**Acceptance Scenarios**:

1. **Given** a cargo value above the insurance limit, **When** the request is evaluated, **Then** the system rejects it with a validation error.
2. **Given** a vehicle that cannot support cargo weight, **When** the request is evaluated, **Then** the system rejects it with a validation error.

---

### User Story 3 - Provide audit-ready output (Priority: P3)

Users receive a quotation response with full reference prices and component breakdown for audit and negotiation.

**Why this priority**: The pricing engine must be explainable and traceable.

**Independent Test**: Can be fully tested by verifying that reference prices and all breakdown fields are present and consistent.

**Acceptance Scenarios**:

1. **Given** a valid quotation, **When** the response is generated, **Then** it includes the KM band used and reference prices.
2. **Given** a valid quotation, **When** the response is generated, **Then** each surcharge and insurance component is itemized.

---

### Edge Cases

- No KM band matches the requested km_total; the system returns a validation error.
- Multiple KM bands match (overlapping ranges); the system returns a validation error.
- quantity_of_vehicles is zero or negative; the system returns a validation error.
- Optional surcharge quantities are negative; the system returns a validation error.
- apply_icms is false; tax amount is zero and not applied.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST select exactly one KM band where km_total is within range; no interpolation is allowed.
- **FR-002**: System MUST select base freight by KM band, vehicle type, and trip structure (delivery only vs. return).
- **FR-003**: System MUST multiply base freight by quantity_of_vehicles before any surcharges.
- **FR-004**: System MUST apply dangerous product and refrigerated surcharges as percentages over base freight.
- **FR-005**: System MUST apply helper, tracking, toll, and diarias using the provided static tables and quantities.
- **FR-006**: System MUST calculate RC-DC, RCTR-C, and GRIS as percentages over cargo_value_nf and enforce insurance_limit.
- **FR-007**: System MUST apply negotiation_percentage to the subtotal before tax and include the adjustment in output.
- **FR-008**: If apply_icms is true, system MUST apply ICMS at 12%; otherwise tax is zero.
- **FR-009**: System MUST return all required output fields, including full breakdowns and reference prices.
- **FR-010**: System MUST raise explicit validation errors for missing mandatory inputs or rule violations with no silent fallbacks.
- **FR-011**: Quotation requests MUST include cargo_weight for vehicle capacity validation.
- **FR-012**: System MUST reject requests when cargo_value_nf exceeds insurance_limit.
- **FR-013**: System MUST round monetary values to 2 decimals at each calculation step.
- **FR-014**: Toll MUST be calculated per vehicle using axle_count × toll_per_axle × quantity_of_vehicles.
- **FR-015**: Diárias MUST be calculated per vehicle using daily_rate × additional_diarias_qty × quantity_of_vehicles.

### Key Entities *(include if feature involves data)*

- **Quotation Request**: Input payload with route, vehicle, trip, cargo, surcharges, commercial parameters, and cargo_weight.
- **Quotation Result**: Output payload with totals, breakdowns, KM band used, and reference prices.
- **KM Band**: Distance range used to select base price for a vehicle type and trip structure.
- **Vehicle Specification**: Static table for max weight, rates, and axle count per vehicle type.
- **Insurance Rule**: Static table of insurance percentages and limits by cargo type.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of valid requests return all required breakdown fields and reference prices.
- **SC-002**: 100% of invalid requests return explicit validation errors without partial totals.
- **SC-003**: Typical quotation calculations complete in under 2 seconds for a single request.
- **SC-004**: Audit review can trace every total back to a single rule or table entry.

## Assumptions

- All static reference tables are provided and versioned externally.
- Currency handling is consistent across all reference tables and inputs.
- Cargo weight data is available in the request or derived by upstream systems.

## Dependencies

- Vehicle Base Table with max weight, daily rate, hourly rate, axle count.
- Additional Charges Table including helper_fixed_value and tracking_fixed_value.
- Cargo Insurance Table with percentages and insurance_limit by cargo type.
- KM Pricing Table with range entries and prices per vehicle type.
