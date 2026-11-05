# Adapted for standalone use in Agent Core.
"""Repair punctuation in complete tool arguments while preserving every token."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from json_repair import repair_json

_TOKEN = re.compile(
    r"""\s+|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|"""
    r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|"
    r"[A-Za-z_][A-Za-z_0-9]*|[{}\[\]:,]"
)


def _tokens(text: str) -> list[tuple[str, Any]]:
    result = []
    position = 0
    for match in _TOKEN.finditer(text):
        if match.start() != position:
            raise ValueError("tool arguments contain incomplete or unsupported tokens")
        position = match.end()
        token = match.group()
        if token.isspace() or token == ",":
            continue
        if token in "{}[]:":
            result.append(("structure", token))
        elif token.startswith(('"', "'")):
            value = (
                json.loads(token) if token.startswith('"') else ast.literal_eval(token)
            )
            result.append(("string", value))
        elif token in ("true", "false", "null"):
            result.append(("literal", token))
        elif token[0].isdigit() or token.startswith("-"):
            result.append(("number", json.loads(token)))
        else:
            result.append(("string", token))
    if position != len(text):
        raise ValueError("tool arguments contain an unfinished token")
    return result


def load_argument_object(raw: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Allow quote/comma repair, never invent values, drop keys, or close truncation."""
    repairs: tuple[str, ...] = ()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        text = raw.strip()
        if not text.startswith("{") or not text.endswith("}"):
            raise ValueError(
                "tool arguments must contain a complete JSON object"
            ) from None
        original_tokens = _tokens(text)
        value = repair_json(text, return_objects=True, skip_json_loads=True)
        if original_tokens != _tokens(json.dumps(value, ensure_ascii=False)):
            raise ValueError("JSON repair would change argument values or structure")
        repairs = ("json_punctuation",)
    if not isinstance(value, dict):
        raise TypeError("tool_call arguments JSON must decode to an object")
    return value, repairs
