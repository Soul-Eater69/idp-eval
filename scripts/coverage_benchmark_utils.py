"""Pure helpers shared by the development-only benchmark scripts.

This module performs no model calls and has no Phoenix dependency. It contains
only transparent normalization, descriptive statistics, and case loading.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def normalize(text: str) -> str:
    """Lowercases and collapses whitespace for exact diagnostics."""
    return " ".join(text.lower().split())


def pairwise_jaccards(sets: list[set[str]]) -> list[float]:
    """Returns all unordered pairwise Jaccard overlaps."""
    overlaps: list[float] = []
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            union = left | right
            overlaps.append(len(left & right) / len(union) if union else 1.0)
    return overlaps


def mean(values: list[float]) -> float | None:
    """Returns the arithmetic mean, or ``None`` for an empty list."""
    return sum(values) / len(values) if values else None


def stddev(values: list[float]) -> float | None:
    """Returns population standard deviation, or ``None`` when empty."""
    if not values:
        return None
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def numeric_stats(values: list[float]) -> dict[str, float | None]:
    """Returns mean, median, min, max, range, and population stddev."""
    if not values:
        return {
            key: None
            for key in ("mean", "median", "min", "max", "range", "stddev")
        }
    return {
        "mean": mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "stddev": stddev(values),
    }


def exact_set_summary(text_runs: list[list[str]]) -> dict[str, Any]:
    """Summarizes count and normalized-exact overlap across text-list runs."""
    counts = [float(len(run)) for run in text_runs]
    sets = [{normalize(text) for text in run} for run in text_runs]
    overlaps = pairwise_jaccards(sets)
    return {
        "count_per_run": [len(run) for run in text_runs],
        "count": numeric_stats(counts),
        "mean_exact_jaccard": mean(overlaps),
        "min_exact_jaccard": min(overlaps) if overlaps else None,
    }


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Loads and validates a JSON array of benchmark cases."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Benchmark cases file must contain a JSON array.")
    required = ("case_id", "input", "context", "output")
    for case in data:
        missing = [field for field in required if field not in case]
        if missing:
            raise ValueError(
                f"Benchmark case {case.get('case_id', '?')!r} missing {missing}."
            )
    return data
