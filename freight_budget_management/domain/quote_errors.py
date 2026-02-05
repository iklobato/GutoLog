"""Error types for freight quote calculations."""

from __future__ import annotations


class QuoteValidationError(ValueError):
    """Raised when a quotation request fails validation."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message
