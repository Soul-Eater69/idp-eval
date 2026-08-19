"""Explicit judge backend constructors."""

from __future__ import annotations

from idp_eval.judges.azure import create_azure_judge
from idp_eval.judges.gateway import create_gateway_judge

__all__ = ["create_azure_judge", "create_gateway_judge"]
