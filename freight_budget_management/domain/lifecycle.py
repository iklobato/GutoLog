"""Lifecycle rules for Freight Budget Management quotations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Set

import yaml

DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[2] / "specs" / "freight_budget_management.spec.yaml"


def load_spec(path: Path | None = None) -> dict:
    """Load the system specification YAML."""
    spec_path = Path(os.getenv("FREIGHT_BUDGET_SPEC_PATH", path or DEFAULT_SPEC_PATH))
    if not spec_path.exists():
        raise FileNotFoundError(f"Specification file not found: {spec_path}")
    with spec_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _get_lifecycle(spec: dict) -> dict:
    if "lifecycle" not in spec:
        raise KeyError("Specification missing required 'lifecycle' section")
    return spec["lifecycle"]


def get_initial_state(spec: dict) -> str:
    lifecycle = _get_lifecycle(spec)
    if "initial_state" not in lifecycle:
        raise KeyError("Specification missing lifecycle.initial_state")
    return lifecycle["initial_state"]


def get_immutable_states(spec: dict) -> Set[str]:
    lifecycle = _get_lifecycle(spec)
    immutable_states = lifecycle.get("immutable_states", [])
    return set(immutable_states)


def get_transitions(spec: dict) -> Dict[str, List[str]]:
    lifecycle = _get_lifecycle(spec)
    transitions = lifecycle.get("transitions")
    if transitions is None:
        raise KeyError("Specification missing lifecycle.transitions")
    if isinstance(transitions, dict):
        return {state: list(next_states) for state, next_states in transitions.items()}
    if isinstance(transitions, list):
        mapping: Dict[str, List[str]] = {}
        for transition in transitions:
            from_state = transition.get("from")
            to_state = transition.get("to")
            if from_state is None or to_state is None:
                raise KeyError("Transition entries must include 'from' and 'to'")
            mapping.setdefault(from_state, []).append(to_state)
        return mapping
    raise TypeError("lifecycle.transitions must be a mapping or list")


def get_states(spec: dict) -> Set[str]:
    lifecycle = _get_lifecycle(spec)
    states: Set[str] = set(lifecycle.get("states", []))
    states.add(get_initial_state(spec))
    states.update(get_immutable_states(spec))
    for from_state, next_states in get_transitions(spec).items():
        states.add(from_state)
        states.update(next_states)
    return states


def is_transition_allowed(spec: dict, current_state: str, next_state: str) -> bool:
    transitions = get_transitions(spec)
    return next_state in transitions.get(current_state, [])


def next_states(spec: dict, current_state: str) -> Iterable[str]:
    return get_transitions(spec).get(current_state, [])


def is_immutable_state(spec: dict, state: str) -> bool:
    return state in get_immutable_states(spec)
