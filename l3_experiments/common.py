"""Small shared helpers for the L3 capability experiment notebooks.

Keep experiment-specific retrieval and prompt construction in the notebooks.
This module only contains identical infrastructure used by every experiment.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx
import pandas as pd
from dotenv import load_dotenv


class IDPGatewayClient:
    """Session-reusable client for the IDP LLM chat-completions gateway."""

    def __init__(self, verify_ssl: bool = False, timeout: float = 90.0):
        load_dotenv()
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.model = self._required_env("LLM_MODEL")
        self.app_id = self._required_env("LLM_APP_ID")
        self.auth_url = self._required_env("IDP_AUTH_URL")
        self.gateway_url = (
            self._required_env("LLM_BASE_URL").rstrip("/")
            + "/api/v1/chatcompletions"
        )
        self._client = httpx.Client(verify=verify_ssl, timeout=timeout)
        self._token: str | None = None

    @staticmethod
    def _required_env(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing environment variable: {name}")
        return value

    def _get_token(self) -> str:
        response = self._client.post(
            self.auth_url,
            headers={
                "Accept": "*/*",
                "ClientId": self._required_env("IDP_CLIENT_ID"),
                "ClientSecret": self._required_env("IDP_CLIENT_SECRET"),
                "scope": "profile openid roles permissions",
            },
            json={
                "username": self._required_env("IDP_USER"),
                "password": self._required_env("IDP_PASSWORD"),
            },
        )
        response.raise_for_status()

        token = response.json().get("jwt_token")
        if not token:
            raise RuntimeError("IDP token response did not contain jwt_token.")
        return token

    def complete(
        self,
        messages: list[dict[str, str]],
        **options: Any,
    ) -> dict[str, Any]:
        """Send chat messages and return the complete gateway response."""
        payload = {
            "model": self.model,
            "messages": messages,
            "api_version": "2024-04-01-preview",
            **options,
        }

        for attempt in range(2):
            if self._token is None:
                self._token = self._get_token()

            response = self._client.post(
                self.gateway_url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "app-id": self.app_id,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=payload,
            )

            if response.status_code != 401 or attempt == 1:
                break

            self._token = self._get_token()

        if response.is_error:
            raise RuntimeError(
                f"IDP gateway request failed ({response.status_code}): {response.text}"
            )

        result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected IDP gateway response: {result}")
        return result

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        **options: Any,
    ) -> str:
        """Return the assistant text for a system prompt and a user prompt."""
        response = self.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **options,
        )

        choice = response.get("choice") or (response.get("choices") or [None])[0]
        content = (choice or {}).get("message", {}).get("content")

        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected IDP gateway response: {response}")
        return content

    def close(self) -> None:
        """Close the persistent HTTP connection when the notebook is finished."""
        self._client.close()


_idp_gateway: IDPGatewayClient | None = None


def get_idp_gateway() -> IDPGatewayClient:
    """Create the IDP LLM gateway client once and reuse it across notebook cells."""
    global _idp_gateway
    if _idp_gateway is None:
        _idp_gateway = IDPGatewayClient(verify_ssl=False)
    return _idp_gateway


def load_gateway() -> IDPGatewayClient:
    """Backward-compatible notebook helper returning the shared IDP gateway."""
    return get_idp_gateway()


def call_llm(gateway: Any, system_prompt: str, user_prompt: str) -> str:
    """Call the IDP gateway using the interface shared by every experiment."""
    response = gateway.chat(system_prompt=system_prompt, user_prompt=user_prompt)
    if not isinstance(response, str):
        raise TypeError(
            f"gateway.chat(...) must return str, got {type(response).__name__}"
        )
    return response


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON from a model response, tolerating a Markdown fence or surrounding text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("LLM returned an empty response.")

    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidate = fenced.group(1) if fenced else text.strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        parsed = None
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                parsed = value
                break
        if parsed is None:
            raise ValueError(f"LLM did not return a JSON object: {text}") from None

    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response must be an object.")
    return parsed


def validate_l3_response(
    payload: Mapping[str, Any],
    candidate_ids: Iterable[str],
    *,
    allow_empty: bool = True,
    max_selected: int = 3,
) -> list[dict[str, str]]:
    """Validate the strict L3 output contract and return normalized selections."""
    if set(payload) != {"l3"}:
        raise ValueError("LLM response must contain exactly one top-level field: l3.")

    raw = payload.get("l3")
    if not isinstance(raw, list):
        raise ValueError("LLM response must contain an 'l3' list.")
    if len(raw) > max_selected:
        raise ValueError(f"LLM may select at most {max_selected} L3 capabilities.")
    if not raw and not allow_empty:
        raise ValueError("LLM returned no L3 selections.")

    allowed = {str(value).strip() for value in candidate_ids if str(value).strip()}
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()

    for index, selection in enumerate(raw, start=1):
        if not isinstance(selection, Mapping):
            raise ValueError(f"L3 selection #{index} must be a JSON object.")
        if set(selection) != {"capability_id", "reason"}:
            raise ValueError(
                f"L3 selection #{index} must contain exactly capability_id and reason."
            )

        capability_id = str(selection.get("capability_id", "")).strip()
        reason = str(selection.get("reason", "")).strip()

        if not capability_id:
            raise ValueError(f"L3 selection #{index} is missing capability_id.")
        if capability_id not in allowed:
            raise ValueError(
                f"LLM selected {capability_id}, which is not a supplied candidate."
            )
        if not reason:
            raise ValueError(f"LLM did not provide a reason for {capability_id}.")
        if capability_id in seen:
            raise ValueError(f"LLM returned duplicate capability_id {capability_id}.")

        seen.add(capability_id)
        normalized.append({"capability_id": capability_id, "reason": reason})

    return normalized


def score_sets(
    predicted: Iterable[str],
    truth: Iterable[str],
) -> dict[str, float | int]:
    """Calculate exact-set match, precision, recall and F1 for one Epic."""
    predicted_set = {str(value).strip() for value in predicted if str(value).strip()}
    truth_set = {str(value).strip() for value in truth if str(value).strip()}

    if not predicted_set and not truth_set:
        precision = recall = f1 = 1.0
    else:
        true_positives = len(predicted_set & truth_set)
        precision = true_positives / len(predicted_set) if predicted_set else 0.0
        recall = true_positives / len(truth_set) if truth_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    return {
        "exact_match": int(predicted_set == truth_set),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_count": len(predicted_set),
        "truth_count": len(truth_set),
    }


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Return a compact one-row summary over rows that have evaluation metrics."""
    if results.empty:
        return pd.DataFrame(
            [
                {
                    "evaluated_epics": 0,
                    "exact_match_accuracy": 0.0,
                    "mean_precision": 0.0,
                    "mean_recall": 0.0,
                    "mean_f1": 0.0,
                    "error_rows": 0,
                }
            ]
        )

    evaluated = results.loc[results["exact_match"].notna()].copy()
    return pd.DataFrame(
        [
            {
                "evaluated_epics": int(len(evaluated)),
                "exact_match_accuracy": float(evaluated["exact_match"].mean())
                if len(evaluated)
                else 0.0,
                "mean_precision": float(evaluated["precision"].mean())
                if len(evaluated)
                else 0.0,
                "mean_recall": float(evaluated["recall"].mean())
                if len(evaluated)
                else 0.0,
                "mean_f1": float(evaluated["f1"].mean()) if len(evaluated) else 0.0,
                "error_rows": int((results["status"] == "error").sum())
                if "status" in results
                else 0,
            }
        ]
    )


def save_results_excel(
    results: pd.DataFrame,
    experiment_name: str,
    output_dir: str | Path = "results",
) -> Path:
    """Save prediction rows and summary to a single Excel workbook."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment_name}.xlsx"
    summary = summarize_results(results)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="predictions", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

        for sheet_name in ("predictions", "summary"):
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                width = min(
                    max(len(str(cell.value or "")) for cell in column_cells) + 2,
                    80,
                )
                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = width

    return output_path
