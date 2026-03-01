# polyglot/scaffold.py
"""
Generate language directories from bins.json metadata.

Reads the MDC bins (language -> track -> dataset_ids) and creates
a langs/<code>/lang.yaml for each language with its MDC sources.
"""

from __future__ import annotations

import json
from pathlib import Path

from polyglot.config import LangConfig, MdcSource, LANGS_ROOT

try:
    import langcodes

    def _lang_name(code: str) -> str:
        code = code.strip().split(",")[0].strip()
        try:
            name = langcodes.Language.get(code).display_name()
            return name if name and name != code else code
        except Exception:
            return code

    def _lang_script(code: str) -> str | None:
        code = code.strip().split(",")[0].strip()
        try:
            s = langcodes.Language.get(code).script_name()
            return s if s else None
        except Exception:
            return None
except ImportError:

    def _lang_name(code: str) -> str:
        return code.strip()

    def _lang_script(code: str) -> str | None:
        return None


def scaffold_from_bins(
    bins_path: Path, lang: str | None = None, overwrite: bool = False
) -> int:
    """Create lang.yaml files from bins.json. Returns count of languages scaffolded."""
    bins = json.loads(bins_path.read_text(encoding="utf-8"))
    created = 0

    for lang_code, tracks in sorted(bins.items()):
        if lang and lang_code != lang:
            continue

        # Skip multi-language codes (e.g. "ewo, fr") — they need manual curation
        if "," in lang_code or " " in lang_code.strip():
            continue

        yaml_path = LANGS_ROOT / lang_code / "lang.yaml"
        if yaml_path.exists() and not overwrite:
            continue

        mdc_sources = []
        for track, ids in tracks.items():
            for did in ids:
                mdc_sources.append(MdcSource(id=did, track=track))

        cfg = LangConfig(
            language=lang_code,
            name=_lang_name(lang_code),
            script=_lang_script(lang_code),
            sources_mdc=mdc_sources,
        )
        cfg.save()
        created += 1

    return created
