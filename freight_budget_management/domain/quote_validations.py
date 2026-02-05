"""Validation rules for freight quotation requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from freight_budget_management.domain.quote_errors import QuoteValidationError
from freight_budget_management.domain.quote_tables import QuoteTables


@dataclass(frozen=True)
class QuoteRequest:
    origin_city: str
    destination_city: str
    km_total: int
    vehicle_type: str
    has_return: bool
    quantity_of_vehicles: int
    cargo_type: str
    cargo_value_nf: float
    cargo_weight: float
    is_dangerous_product: bool
    is_refrigerated: bool
    needs_helper: bool
    needs_tracking_device: bool
    additional_diarias_qty: int
    toll_per_axle: float
    negotiation_percentage: float
    apply_icms: bool


def validate_request(request: QuoteRequest, tables: QuoteTables) -> None:
    _require_positive(request.quantity_of_vehicles, "quantity_of_vehicles")
    _require_non_negative(request.additional_diarias_qty, "additional_diarias_qty")
    _require_non_negative(request.toll_per_axle, "toll_per_axle")
    _require_non_negative(request.cargo_value_nf, "cargo_value_nf")
    _require_non_negative(request.cargo_weight, "cargo_weight")
    _require_non_negative(request.negotiation_percentage, "negotiation_percentage")

    if request.vehicle_type not in tables.vehicles:
        raise QuoteValidationError("vehicle_type", "Unsupported vehicle type")
    if request.cargo_type not in tables.insurance_rules:
        raise QuoteValidationError("cargo_type", "Unsupported cargo type")

    vehicle = tables.vehicles[request.vehicle_type]
    if request.cargo_weight > vehicle.max_weight:
        raise QuoteValidationError("cargo_weight", "Cargo exceeds vehicle max weight")

    insurance = tables.insurance_rules[request.cargo_type]
    if request.cargo_value_nf > insurance.insurance_limit:
        raise QuoteValidationError("cargo_value_nf", "Cargo value exceeds insurance limit")


def validate_km_band_match(km_total: int, bands: Iterable) -> None:
    matches = [band for band in bands if band.km_range_start <= km_total <= band.km_range_end]
    if len(matches) == 0:
        raise QuoteValidationError("km_total", "No KM band matches requested distance")
    if len(matches) > 1:
        raise QuoteValidationError("km_total", "Multiple KM bands match requested distance")


def _require_positive(value: int, field: str) -> None:
    if value <= 0:
        raise QuoteValidationError(field, "Value must be greater than zero")


def _require_non_negative(value: float | int, field: str) -> None:
    if value < 0:
        raise QuoteValidationError(field, "Value must be zero or greater")
