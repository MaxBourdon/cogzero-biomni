# canonicalizer.py

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Optional


# Match floats with exactly one decimal point and at least one digit on both sides.
# Do NOT match if a letter touches the number on either side (e.g., "v2.0", "2.0alpha").
_FLOAT_RE = re.compile(
    r"""
    (?<![A-Za-z0-9.])          # NOT preceded by letter/digit/dot
    (?P<num>[+-]?\d+\.\d+)     # float with mandatory decimal part
    (?![A-Za-z0-9.])           # NOT followed by letter/digit/dot
    """,
    re.VERBOSE,
)


@dataclass
class CanonSettings:
    decimals: int = 4
    normalize_quotes: bool = True
    normalize_null: str = "None"
    normalize_booleans: bool = True
    collapse_whitespace: bool = True


def _normalize_whitespace(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_quotes(s: str) -> str:
    return s.replace('"', "'")


def _normalize_nulls(s: str, null_token: str) -> str:
    return re.sub(r"\b(?:None|null|NULL|NaN|nan)\b", null_token, s)


def _normalize_booleans(s: str) -> str:
    s = re.sub(r"\btrue\b", "True", s, flags=re.IGNORECASE)
    s = re.sub(r"\bfalse\b", "False", s, flags=re.IGNORECASE)
    return s


def _round_float_match(m: re.Match, decimals: int) -> str:
    txt = m.group("num")
    try:
        val = float(txt)
    except ValueError:
        return txt
    return f"{val:.{decimals}f}"


def canonicalize_output(s: str, settings: Optional[CanonSettings] = None) -> str:
    """
    Simple canonicalization:
      - convert all double quotes to single quotes
      - normalize null-like tokens
      - normalize booleans (true/false -> True/False)
      - round only floats (integers unchanged), skipping numbers adjacent to letters
      - collapse whitespace
    """
    settings = settings or CanonSettings()
    s = str(s)

    if settings.normalize_quotes:
        s = _normalize_quotes(s)

    s = _normalize_nulls(s, settings.normalize_null)
    if settings.normalize_booleans:
        s = _normalize_booleans(s)

    # Round floats only
    s = _FLOAT_RE.sub(lambda m: _round_float_match(m, settings.decimals), s)

    if settings.collapse_whitespace:
        s = _normalize_whitespace(s)

    return s


def compare_outputs(pred: str, gt: str, settings: Optional[CanonSettings] = None) -> dict:
    settings = settings or CanonSettings()
    pred_c = canonicalize_output(pred, settings)
    gt_c = canonicalize_output(gt, settings)
    equal = pred_c == gt_c

    diff = ""
    if not equal:
        sm = difflib.SequenceMatcher(a=gt_c, b=pred_c)
        diff = f"\n- GT  : {gt_c}\n+ Pred: {pred_c}\nΔ mismatch ~{int((1 - sm.ratio()) * 100)}%"

    return {"equal": equal, "pred_canon": pred_c, "gt_canon": gt_c, "diff": diff}
