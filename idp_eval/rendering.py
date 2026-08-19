"""Deterministic rendering of generic structured evaluation values."""

from __future__ import annotations

from typing import Any

StructuredValue = Any  # str | int | float | bool | None | dict[str, ...] | list[...]

_INDENT = "  "


def _is_scalar(value: Any) -> bool:
    """True for the leaf value types (``None`` counts as a scalar)."""
    return value is None or isinstance(value, (str, int, float, bool))


def _scalar_text(value: Any) -> str:
    """Renders a leaf value predictably; strings pass through unchanged."""
    if value is None:
        return ""
    if isinstance(value, bool):  # before int: bool is an int subclass
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return str(value)  # int / float


def _humanize(key: str) -> str:
    """Turns a ``snake_case`` key into a readable label.

    Already-uppercase tokens are preserved; other tokens are capitalized.
    """
    parts = [part for part in str(key).split("_") if part]
    return " ".join(
        part if part.isupper() else part.capitalize() for part in parts
    )


def validate_structured_value(value: Any, path: str = "value") -> None:
    """Validates that ``value`` is a supported (recursively) structured value.

    Supported: ``str``, ``int``, ``float``, ``bool``, ``None``, ``dict`` with
    string keys, and ``list``. Anything else (sets, bytes, tuples, arbitrary
    objects, non-string dict keys) raises ``TypeError`` naming the offending
    path, so failures surface at case construction — never later inside a prompt.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Unsupported dict key at {path}: keys must be strings, got "
                    f"{type(key).__name__}."
                )
            validate_structured_value(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_structured_value(item, f"{path}[{index}]")
        return
    raise TypeError(
        f"Unsupported value type at {path}: {type(value).__name__}. Supported "
        "types are str, int, float, bool, None, dict, and list."
    )


def is_empty_value(value: Any) -> bool:
    """Shared notion of "missing / empty" required evaluation content.

    Empty: ``None``, an empty/whitespace-only string, an empty ``dict``, or an
    empty ``list``. Scalars such as ``0`` and ``False`` are legitimate values and
    are **not** treated as empty — required-field checks decide separately whether
    a metric needs textual content.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list)):
        return len(value) == 0
    return False  # int / float / bool are present values


def _render_lines(value: Any, indent: int) -> list[str]:
    """Recursively renders ``value`` to indented lines (no trailing newline)."""
    pad = _INDENT * indent

    if _is_scalar(value):
        return [pad + line for line in _scalar_text(value).split("\n")]

    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            if _is_scalar(item):
                lines.append(pad + "- " + _scalar_text(item))
            else:
                # Put the first nested line on the bullet to keep list-of-dict
                # output compact while preserving the remaining indentation.
                block = _render_lines(item, indent + 1)
                if block:
                    lines.append(pad + "- " + block[0].lstrip())
                    lines.extend(block[1:])
                else:  # empty nested dict/list
                    lines.append(pad + "-")
        return lines

    # A blank line separates dictionary fields without changing their order.
    blocks: list[list[str]] = []
    for key, item in value.items():
        label = _humanize(key)
        if _is_scalar(item):
            text = _scalar_text(item)
            if text == "":
                block = [pad + label + ":"]
            else:
                text_lines = text.split("\n")
                block = [pad + f"{label}: {text_lines[0]}"] + [
                    pad + line for line in text_lines[1:]
                ]
        elif isinstance(item, list):
            block = [pad + label + ":"]
            block.extend(_render_lines(item, indent))
        else:
            block = [pad + label + ":"]
            block.extend(_render_lines(item, indent + 1))
        blocks.append(block)

    out: list[str] = []
    for index, block in enumerate(blocks):
        if index:
            out.append("")
        out.extend(block)
    return out


def render_value(value: StructuredValue) -> str:
    """Renders a structured value to readable, judge-friendly text.

    Strings pass through unchanged. Dicts become ``Label:``-headed blocks in
    insertion order; scalar lists become bullet lists; nested structures stay
    hierarchical. Empty dicts/lists and ``None`` render to an empty string. The
    result is deterministic and contains no JSON unless a raw string held it.
    """
    validate_structured_value(value)
    return "\n".join(_render_lines(value, 0))
