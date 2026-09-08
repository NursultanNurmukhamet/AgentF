"""Recover JSON from model output that used Python literals, trailing commas or code fences.

WHY THIS EXISTS. Your agent asks the model for a JSON array of commands, and most of the
time it obliges. But models are trained on a lot of Python, and they will sometimes hand
you Python's spelling of a boolean instead of JSON's:

    [{"commandType": "MOVE_TO", "parameters": {"target_x": 2.0, "sprint": True}}]
                                                                        ^^^^

``True`` is valid Python and invalid JSON, so ``json.loads`` raises and the whole command
is thrown away. Your agent then quietly falls back to its rule-based logic. Nothing
crashes and nothing looks broken — the agent simply stops using its model, and the only
clue is a "LLM parse failed" line in CloudWatch. That happened for a full match before
anyone noticed.

So: after a strict parse fails, try once more on a normalised copy. Valid JSON never
reaches this code, so well-behaved output keeps its behaviour exactly.

What is recovered, and nothing else:
  * bare ``True`` / ``False`` / ``None`` tokens outside strings -> ``true`` / ``false`` / ``null``
  * a trailing comma before ``}`` or ``]``
  * a markdown code fence wrapping the payload

Single-quoted strings are deliberately NOT rewritten: apostrophes inside prose make that
guess unsafe, and a single-quoted payload never parsed before either.

Note what is NOT touched: anything inside a string. ``{"note": "True story"}`` comes back
unchanged, because rewriting a model's prose would be worse than dropping a command.
"""

import json
import re

_PY_LITERALS = (("True", "true"), ("False", "false"), ("None", "null"))

_WORD_CHAR = re.compile(r"[A-Za-z0-9_]")

_FENCED_BLOCK = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n?([\s\S]*?)```")


def _is_word_char(ch: str) -> bool:
    return bool(ch) and bool(_WORD_CHAR.match(ch))


def _strip_fences(text: str) -> str:
    """Return the contents of the first markdown code fence, or the text unchanged."""
    if "```" not in text:
        return text
    match = _FENCED_BLOCK.search(text)
    if match and match.group(1).strip():
        return match.group(1)
    return text.replace("```", " ")


def normalise_json_text(text: str) -> str:
    """Rewrite Python-flavoured JSON into JSON, leaving string contents untouched."""
    src = _strip_fences(text)
    out = []
    i = 0
    n = len(src)
    in_string = False
    while i < n:
        ch = src[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(src[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        replaced = False
        for py, js in _PY_LITERALS:
            if not src.startswith(py, i):
                continue
            before = src[i - 1] if i > 0 else ""
            after = src[i + len(py)] if i + len(py) < n else ""
            if _is_word_char(before) or _is_word_char(after):
                continue
            out.append(js)
            i += len(py)
            replaced = True
            break
        if replaced:
            continue

        if ch == ",":
            j = i + 1
            while j < n and src[j].isspace():
                j += 1
            if j < n and src[j] in "}]":
                i += 1  # drop the trailing comma
                continue

        out.append(ch)
        i += 1
    return "".join(out)


def parse_json_tolerant(text: str):
    """Parse ``text`` as JSON, retrying once on a normalised copy.

    Returns ``(value, recovered)`` where ``recovered`` is True only when the strict parse
    failed and the normalised retry succeeded, or ``None`` when nothing parses.
    """
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass

    normalised = normalise_json_text(text)
    if normalised == text:
        return None
    try:
        return json.loads(normalised), True
    except json.JSONDecodeError:
        return None
