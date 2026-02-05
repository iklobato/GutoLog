"""Static table structures and loaders for freight quote engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass(frozen=True)
class VehicleSpec:
    vehicle_type: str
    max_weight: float
    daily_rate: float
    hourly_rate: float
    axle_count: int


@dataclass(frozen=True)
class AdditionalCharges:
    dangerous_product_percent: float
    refrigerated_percent: float
    helper_fixed_value: float
    tracking_fixed_value: float


@dataclass(frozen=True)
class InsuranceRule:
    cargo_type: str
    rc_dc_percent: float
    rctrc_percent: float
    gris_percent: float
    insurance_limit: float


@dataclass(frozen=True)
class KmBand:
    km_range_start: int
    km_range_end: int
    price_per_vehicle_type: Dict[str, float]
    delivery_price: float
    return_price: float
    reference_price: float


@dataclass(frozen=True)
class QuoteTables:
    vehicles: Dict[str, VehicleSpec]
    additional_charges: AdditionalCharges
    insurance_rules: Dict[str, InsuranceRule]
    km_bands: List[KmBand]


def load_tables(path: Path) -> QuoteTables:
    data = _load_yaml(path)
    vehicles = {
        item["vehicle_type"]: VehicleSpec(**item)
        for item in data["vehicle_base_table"]
    }
    additional = AdditionalCharges(**data["additional_charges"])
    insurance = {
        item["cargo_type"]: InsuranceRule(**item)
        for item in data["cargo_insurance_table"]
    }
    km_bands = [KmBand(**item) for item in data["km_pricing_table"]]
    return QuoteTables(
        vehicles=vehicles,
        additional_charges=additional,
        insurance_rules=insurance,
        km_bands=km_bands,
    )


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data
