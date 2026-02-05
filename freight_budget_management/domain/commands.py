"""Command definitions for Freight Budget Management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from freight_budget_management.domain.lifecycle import load_spec


@dataclass(frozen=True)
class CommandDefinition:
    name: str
    allowed_states: Optional[Iterable[str]]
    required_fields: Iterable[str]
    raw: Dict[str, Any]


def _get_commands_section(spec: dict) -> dict | list:
    if "commands" not in spec:
        raise KeyError("Specification missing required 'commands' section")
    return spec["commands"]


def list_command_definitions(spec: dict | None = None) -> Dict[str, CommandDefinition]:
    spec = spec or load_spec()
    commands_section = _get_commands_section(spec)
    definitions: Dict[str, CommandDefinition] = {}

    if isinstance(commands_section, dict):
        for name, command in commands_section.items():
            allowed_states = command.get("allowed_states") or command.get("valid_from") or command.get("from_states")
            required_fields = command.get("required_fields") or command.get("required") or []
            definitions[name] = CommandDefinition(
                name=name,
                allowed_states=allowed_states,
                required_fields=required_fields,
                raw=command,
            )
        return definitions

    if isinstance(commands_section, list):
        for command in commands_section:
            name = command.get("name")
            if not name:
                raise KeyError("Command entry missing 'name'")
            allowed_states = command.get("allowed_states") or command.get("valid_from") or command.get("from_states")
            required_fields = command.get("required_fields") or command.get("required") or []
            definitions[name] = CommandDefinition(
                name=name,
                allowed_states=allowed_states,
                required_fields=required_fields,
                raw=command,
            )
        return definitions

    raise TypeError("commands section must be a mapping or list")


def get_command_definition(command_name: str, spec: dict | None = None) -> CommandDefinition:
    definitions = list_command_definitions(spec)
    if command_name not in definitions:
        raise KeyError(f"Unknown command: {command_name}")
    return definitions[command_name]
