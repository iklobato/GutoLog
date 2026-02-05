# Data Model: Freight Quote Engine

## Entity: Quotation Request
- **Purpose**: Input payload for quotation calculation.
- **Fields**:
  - `origin_city`
  - `destination_city`
  - `km_total`
  - `vehicle_type`
  - `has_return`
  - `quantity_of_vehicles`
  - `cargo_type`
  - `cargo_value_nf`
  - `cargo_weight`
  - `is_dangerous_product`
  - `is_refrigerated`
  - `needs_helper`
  - `needs_tracking_device`
  - `additional_diarias_qty`
  - `toll_per_axle`
  - `negotiation_percentage`
  - `apply_icms`

## Entity: Quotation Result
- **Purpose**: Output payload for audit-ready pricing.
- **Fields**:
  - `km_band_used`
  - `vehicle_type`
  - `freight_base`
  - `surcharges_breakdown`
  - `insurance_breakdown`
  - `negotiation_adjustment`
  - `tax_amount`
  - `final_total`
  - `reference_prices`

## Entity: Vehicle Specification
- **Purpose**: Static table for vehicle limits and rates.
- **Fields**:
  - `vehicle_type`
  - `max_weight`
  - `daily_rate`
  - `hourly_rate`
  - `axle_count`

## Entity: Additional Charges
- **Purpose**: Static surcharge configuration.
- **Fields**:
  - `dangerous_product_percent`
  - `refrigerated_percent`
  - `helper_fixed_value`
  - `tracking_fixed_value`

## Entity: Insurance Rule
- **Purpose**: Static insurance configuration per cargo type.
- **Fields**:
  - `cargo_type`
  - `rc_dc_percent`
  - `rctrc_percent`
  - `gris_percent`
  - `insurance_limit`

## Entity: KM Band
- **Purpose**: Static pricing table per distance band and vehicle.
- **Fields**:
  - `km_range_start`
  - `km_range_end`
  - `price_per_vehicle_type`
  - `delivery_price`
  - `return_price`
  - `reference_price`

## Validation Rules
- `km_total` must match exactly one KM band.
- `cargo_value_nf` must be <= insurance_limit for the cargo_type.
- `cargo_weight` must be <= max_weight for the vehicle_type.
- `quantity_of_vehicles` and surcharge quantities must be >= 0.
- Monetary values rounded to 2 decimals at each step.
