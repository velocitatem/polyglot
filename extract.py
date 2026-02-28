#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Split columns by either:
#  - one-or-more tabs, OR
#  - 2+ normal spaces, OR
#  - 2+ non-breaking spaces (just in case)
SEP_RE = re.compile(r"(?:\t+| {2,}|\u00A0{2,})")

# ---------- language heuristics ----------

ISO_639_3_RE = re.compile(r"\bISO\s*639-3\s*[:=]\s*([a-z]{3})\b", re.IGNORECASE)
ISO_639_3_ALT_RE = re.compile(r"\bISO\s*639-3\s*[:=]?\s*([a-z]{3})\b", re.IGNORECASE)

DESC_LANGUAGE_RE = re.compile(r"\bLanguage\s*:\s*([A-Za-z][A-Za-z \-’'()]+)", re.IGNORECASE)
IN_X_RE = re.compile(r"\b(?:in|presented in|primarily in)\s+([A-Za-z][A-Za-z \-’']+)", re.IGNORECASE)
FOR_X_RE = re.compile(r"\b(?:for|in)\s+(?:the\s+)?([A-Za-z][A-Za-z \-’']+)\s+language\b", re.IGNORECASE)

PARALLEL_RE = re.compile(r"^(.+?)[-–](.+?)\s+Parallel\s+Corpus", re.IGNORECASE)
COMMON_VOICE_LANG_RE = re.compile(r"^Common Voice .*?-\s*(.+?)\s*$", re.IGNORECASE)

TTS_FOR_RE = re.compile(r"\btext to speech dataset for\s+([A-Za-z][A-Za-z \-’']+)\b", re.IGNORECASE)
ASR_FOR_RE = re.compile(r"\b(?:ASR|speech recognition)\s+(?:for|in)\s+([A-Za-z][A-Za-z \-’']+)\b", re.IGNORECASE)

# Keywords that strongly suggest "bundle/multilingual"
MULTI_HINTS = (
    "multilingual",
    "bundle",
    "across",
    "40 languages",
    "300 languages",
    "language identification",
    "cross-lingual",
    "shared task",
)

# Tokens that suggest a dataset name starts with a language
DATASET_TAIL_MARKERS = (
    "corpus",
    "dataset",
    "bench",
    "speech",
    "tts",
    "asr",
    "ner",
    "nli",
    "newspaper",
    "magazine",
    "literature",
    "parallel",
)

def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\u00A0", " ")).strip()

def _strip_trailing_code(s: str) -> str:
    # "Khmer (khm)" -> "Khmer"
    return re.sub(r"\s*\([a-z]{2,3}\)\s*$", "", s, flags=re.IGNORECASE).strip()

def infer_primary_language(name: str, desc: str) -> Tuple[str, Optional[str], str]:
    """
    Returns: (primary_language, dialect_or_variant, note)
    """
    n = _norm_spaces(name)
    d = _norm_spaces(desc)
    low = (n + " " + d).lower()

    # Multilingual / bundle
    if any(h in low for h in MULTI_HINTS):
        return ("Multilingual", None, "multilingual_or_bundle_hint")

    # Common Voice: "... - Czech"
    m = COMMON_VOICE_LANG_RE.match(n)
    if m and n.lower().startswith("common voice"):
        lang = _norm_spaces(m.group(1))
        # Some CV entries are bundles like "Single Word Target Segment"
        if "single word" in lang.lower() or "segment" in lang.lower():
            return ("Multilingual", None, "common_voice_bundle_like")
        return (_strip_trailing_code(lang), None, "common_voice_name")

    # Parallel corpus: "Ewondo-French Parallel Corpus"
    m = PARALLEL_RE.match(n)
    if m:
        # For grouping, pick the first as "primary" but keep both in note.
        a = _strip_trailing_code(_norm_spaces(m.group(1)))
        b = _strip_trailing_code(_norm_spaces(m.group(2)))
        return (a, None, f"parallel_corpus_primary={a}_secondary={b}")

    # Explicit "Language: X"
    m = DESC_LANGUAGE_RE.search(d)
    if m:
        lang = _strip_trailing_code(_norm_spaces(m.group(1)))
        return (lang, None, "desc_language_field")

    # "text to speech dataset for X"
    m = TTS_FOR_RE.search(n) or TTS_FOR_RE.search(d)
    if m:
        lang = _strip_trailing_code(_norm_spaces(m.group(1)))
        return (lang, None, "tts_for")

    # "presented in Indonesian", "primarily in X"
    m = IN_X_RE.search(d)
    if m:
        lang = _strip_trailing_code(_norm_spaces(m.group(1)))
        # avoid grabbing "the year 2022" type false positives
        if len(lang.split()) <= 5 and not any(ch.isdigit() for ch in lang):
            return (lang, None, "desc_in_language_phrase")

    # "for the X language"
    m = FOR_X_RE.search(d)
    if m:
        lang = _strip_trailing_code(_norm_spaces(m.group(1)))
        return (lang, None, "desc_for_language_phrase")

    # Name-based: "<Language> ... Corpus/Dataset/..."
    # e.g. "Saraiki Literature Corpus" -> Saraiki
    tokens = n.split()
    if tokens:
        # If the name ends with a known language family word (e.g. "Nahuatl")
        # and the title is short, treat last token as primary language and rest as dialect/variant.
        if len(tokens) <= 4 and tokens[-1][0].isupper():
            # Heuristic: if title doesn't contain common dataset markers, still accept "Tetelancingo Nahuatl"
            # as (Nahuatl, Tetelancingo)
            primary = tokens[-1]
            dialect = " ".join(tokens[:-1]).strip() or None
            # Only do this if it "looks like a language label" (capitalized) and not obviously something else
            return (primary, dialect, "name_short_last_token_primary")

        # If marker exists, assume the language is the prefix up to first marker-ish token
        lower_tokens = [t.lower() for t in tokens]
        for i, t in enumerate(lower_tokens):
            if t in DATASET_TAIL_MARKERS and i > 0:
                lang = " ".join(tokens[:i]).strip()
                if 1 <= len(lang.split()) <= 5:
                    return (_strip_trailing_code(lang), None, "name_prefix_before_marker")

    return ("Unknown", None, "unresolved")

# ---------- parsing ----------

def parse_line_to_fields(line: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse a single line into (id, name, desc).
    Accepts:
      - true TSV with tabs
      - columnar spacing with 2+ spaces
    """
    raw = line.rstrip("\n").replace("\u00A0", " ")
    s = raw.strip()
    if not s:
        return None
    if s.startswith("----"):
        return None
    # Skip JSON-looking content if present
    if s[0] in "{[":
        return None

    parts = SEP_RE.split(s, maxsplit=2)
    if len(parts) >= 2:
        ds_id = parts[0].strip()
        name = parts[1].strip()
        desc = parts[2].strip() if len(parts) > 2 else ""
        return (ds_id, name, desc)

    # If no split happened, treat as failure
    return None

def read_rows(path: Path) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    rows: List[Tuple[str, str, str]] = []
    failures: List[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_line_to_fields(line)
            if parsed is None:
                # keep non-empty lines as failures for debugging
                if line.strip() and not line.strip().startswith("----"):
                    failures.append(line.rstrip("\n"))
                continue
            ds_id, name, desc = parsed
            if ds_id.lower() in {"id", "dataset_id"}:
                continue
            rows.append((ds_id, name, desc))

    return rows, failures

# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path", type=Path)
    ap.add_argument("--out", type=Path, default=Path("datasets_by_language.json"))
    ap.add_argument("--failures-out", type=Path, default=Path("parse_failures.txt"))
    ap.add_argument("--drop-unknown", action="store_true")
    args = ap.parse_args()

    rows, failures = read_rows(args.input_path)

    grouped: Dict[str, List[dict]] = defaultdict(list)
    unknown_count = 0

    for ds_id, name, desc in rows:
        primary, dialect, note = infer_primary_language(name, desc)
        entry = {
            "id": ds_id,
            "name": name,
            "desc": desc,
            "primary_language": primary,
            "dialect_or_variant": dialect,
            "note": note,
        }
        grouped[primary].append(entry)
        if primary == "Unknown":
            unknown_count += 1

    if args.drop_unknown:
        grouped.pop("Unknown", None)

    # Stable ordering
    grouped_sorted = {
        lang: sorted(items, key=lambda e: e["id"])
        for lang, items in sorted(grouped.items(), key=lambda kv: kv[0].lower())
    }

    args.out.write_text(json.dumps(grouped_sorted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if failures:
        args.failures_out.write_text("\n".join(failures) + "\n", encoding="utf-8")
    else:
        # ensure old file doesn't mislead
        args.failures_out.write_text("", encoding="utf-8")

    print(f"Wrote {args.out} with {len(grouped_sorted)} language groups from {len(rows)} rows.")
    print(f"Unknown rows: {unknown_count}/{len(rows)}")
    print(f"Parse failures: {len(failures)} (see {args.failures_out})")

if __name__ == "__main__":
    main()
