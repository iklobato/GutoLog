"""Freight quotation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from freight_budget_management.domain.quote_rounding import round_currency, to_decimal
from freight_budget_management.domain.quote_tables import QuoteTables
from freight_budget_management.domain.quote_errors import QuoteValidationError
from freight_budget_management.domain.quote_validations import QuoteRequest, validate_km_band_match, validate_request


@dataclass(frozen=True)
class QuoteResult:
    km_band_used: Dict
    vehicle_type: str
    freight_base: float
    surcharges_breakdown: Dict[str, float]
    insurance_breakdown: Dict[str, float]
    negotiation_adjustment: float
    tax_amount: float
    final_total: float
    reference_prices: Dict[str, float]


def calculate_quote(request: QuoteRequest, tables: QuoteTables) -> QuoteResult:
    validate_request(request, tables)
    validate_km_band_match(request.km_total, tables.km_bands)

    km_band = _select_km_band(request.km_total, tables)
    base_price = _select_base_price(request, km_band)
    freight_base = round_currency(base_price * request.quantity_of_vehicles)

    surcharges = _calculate_surcharges(request, tables, freight_base)
    insurance = _calculate_insurance(request, tables)
    subtotal = round_currency(freight_base + surcharges["total"] + insurance["total"])

    negotiation_adjustment = round_currency(subtotal * to_decimal(request.negotiation_percentage) / to_decimal("100"))
    subtotal_after_negotiation = round_currency(subtotal + negotiation_adjustment)

    tax_amount = round_currency(subtotal_after_negotiation * to_decimal("0.12")) if request.apply_icms else round_currency(0)
    final_total = round_currency(subtotal_after_negotiation + tax_amount)

    return QuoteResult(
        km_band_used=_band_summary(km_band),
        vehicle_type=request.vehicle_type,
        freight_base=float(freight_base),
        surcharges_breakdown={**surcharges, "total": float(surcharges["total"])},
        insurance_breakdown={**insurance, "total": float(insurance["total"])},
        negotiation_adjustment=float(negotiation_adjustment),
        tax_amount=float(tax_amount),
        final_total=float(final_total),
        reference_prices=_reference_prices(km_band),
    )


def _select_km_band(km_total: int, tables: QuoteTables):
    for band in tables.km_bands:
        if band.km_range_start <= km_total <= band.km_range_end:
            return band
    raise ValueError("No KM band found for requested distance")


def _select_base_price(request: QuoteRequest, km_band) -> float:
    if request.vehicle_type not in km_band.price_per_vehicle_type:
        raise QuoteValidationError("vehicle_type", "Vehicle type not priced in selected KM band")
    if request.has_return:
        return km_band.return_price
    return km_band.delivery_price


def _calculate_surcharges(request: QuoteRequest, tables: QuoteTables, freight_base) -> Dict[str, float]:
    charges = tables.additional_charges
    dangerous = round_currency(freight_base * to_decimal(charges.dangerous_product_percent) / to_decimal("100")) if request.is_dangerous_product else round_currency(0)
    refrigerated = round_currency(freight_base * to_decimal(charges.refrigerated_percent) / to_decimal("100")) if request.is_refrigerated else round_currency(0)
    helper = round_currency(charges.helper_fixed_value if request.needs_helper else 0)
    tracking = round_currency(charges.tracking_fixed_value if request.needs_tracking_device else 0)
    vehicle = tables.vehicles[request.vehicle_type]
    toll = round_currency(to_decimal(vehicle.axle_count) * to_decimal(request.toll_per_axle) * to_decimal(request.quantity_of_vehicles))
    diarias = round_currency(to_decimal(vehicle.daily_rate) * to_decimal(request.additional_diarias_qty) * to_decimal(request.quantity_of_vehicles))

    total = round_currency(dangerous + refrigerated + helper + tracking + toll + diarias)
    return {
        "dangerous_product": float(dangerous),
        "refrigerated": float(refrigerated),
        "helper": float(helper),
        "tracking_device": float(tracking),
        "toll": float(toll),
        "diarias": float(diarias),
        "total": total,
    }


def _calculate_insurance(request: QuoteRequest, tables: QuoteTables) -> Dict[str, float]:
    rule = tables.insurance_rules[request.cargo_type]
    base = to_decimal(request.cargo_value_nf)
    rc_dc = round_currency(base * to_decimal(rule.rc_dc_percent) / to_decimal("100"))
    rctrc = round_currency(base * to_decimal(rule.rctrc_percent) / to_decimal("100"))
    gris = round_currency(base * to_decimal(rule.gris_percent) / to_decimal("100"))
    total = round_currency(rc_dc + rctrc + gris)
    return {
        "rc_dc": float(rc_dc),
        "rctrc": float(rctrc),
        "gris": float(gris),
        "total": total,
    }


def _band_summary(band) -> Dict[str, float]:
    return {
        "km_range_start": band.km_range_start,
        "km_range_end": band.km_range_end,
        "delivery_price": band.delivery_price,
        "return_price": band.return_price,
        "reference_price": band.reference_price,
    }


def _reference_prices(band) -> Dict[str, float]:
    return {
        "delivery_price": band.delivery_price,
        "return_price": band.return_price,
        "reference_price": band.reference_price,
    }
