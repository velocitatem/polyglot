from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class PlanRow:
    lang: str
    track: str
    dataset_count: int | None
    train_docs: int | None
    valid_docs: int | None
    reason: str


def _base_key(base_model: str) -> str:
    return base_model.replace("/", "_")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_pairs(bins: dict, track: str | None) -> list[tuple[str, str, int]]:
    pairs: list[tuple[str, str, int]] = []
    for lang, tracks in bins.items():
        for tr, ids in tracks.items():
            if track and tr != track:
                continue
            pairs.append((lang, tr, len(ids)))
    return sorted(pairs, key=lambda x: (-x[2], x[0], x[1]))


def _pick_rows(
    bins: dict,
    coverage: dict | None,
    base_model: str,
    track: str | None,
    min_train_docs: int,
    min_valid_docs: int,
    skip_trained: bool,
) -> list[PlanRow]:
    rows: list[PlanRow] = []
    bk = _base_key(base_model)

    for lang, tr, dataset_count in _iter_pairs(bins, track):
        cov_row = None
        if coverage:
            cov_row = coverage.get("coverage", {}).get(lang, {}).get(tr, {})

        trained_for_base = bool(cov_row and cov_row.get("trained_runs", {}).get(bk))
        if skip_trained and trained_for_base:
            continue

        train_docs = cov_row.get("train_docs") if cov_row else None
        valid_docs = cov_row.get("valid_docs") if cov_row else None

        if train_docs is None:
            reason = "build_data"
        elif train_docs < min_train_docs:
            reason = "train_docs_below_threshold"
        elif valid_docs is None:
            reason = "compute_stats"
        elif valid_docs < min_valid_docs:
            reason = "valid_docs_below_threshold"
        else:
            reason = "train_ready"

        rows.append(
            PlanRow(
                lang=lang,
                track=tr,
                dataset_count=dataset_count,
                train_docs=train_docs,
                valid_docs=valid_docs,
                reason=reason,
            )
        )

    return rows


def _write_shell_script(
    out: Path,
    rows: list[PlanRow],
    base_model: str,
    run_tag_prefix: str,
    hf_repo_prefix: str | None,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("#!/usr/bin/env bash")
    lines.append("set -euo pipefail")
    lines.append("")
    lines.append(f'BASE_MODEL="{base_model}"')
    lines.append("")

    for r in rows:
        run_tag = f"{run_tag_prefix}-{r.lang}-{r.track}"
        lines.append(f"# {r.lang}/{r.track} :: {r.reason}")
        lines.append(f"make data LANG={r.lang} TRACK={r.track}")
        lines.append(
            "make train "
            f'LANG={r.lang} TRACK={r.track} BASE_MODEL="$BASE_MODEL" RUN_TAG={run_tag}'
        )
        lines.append(
            "make eval "
            f'LANG={r.lang} TRACK={r.track} BASE_MODEL="$BASE_MODEL" RUN_TAG={run_tag}'
        )
        if hf_repo_prefix:
            repo = f"{hf_repo_prefix}-{r.lang}-{r.track}"
            lines.append(
                "make publish "
                f'LANG={r.lang} TRACK={r.track} BASE_MODEL="$BASE_MODEL" RUN_TAG={run_tag} '
                f"HF_REPO={repo}"
            )
        lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.chmod(0o755)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create full LoRA adapter training plan")
    ap.add_argument("--bins", type=Path, required=True)
    ap.add_argument("--coverage", type=Path, default=None)
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--track", type=str, default=None)
    ap.add_argument("--min-train-docs", type=int, default=200)
    ap.add_argument("--min-valid-docs", type=int, default=10)
    ap.add_argument("--skip-trained", action="store_true")
    ap.add_argument("--run-tag-prefix", type=str, default="all")
    ap.add_argument("--hf-repo-prefix", type=str, default=None)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-shell", type=Path, required=True)
    a = ap.parse_args()

    bins = _load_json(a.bins)
    coverage = _load_json(a.coverage) if a.coverage and a.coverage.exists() else None

    rows = _pick_rows(
        bins=bins,
        coverage=coverage,
        base_model=a.base_model,
        track=a.track,
        min_train_docs=a.min_train_docs,
        min_valid_docs=a.min_valid_docs,
        skip_trained=a.skip_trained,
    )

    result = {
        "base_model": a.base_model,
        "track_filter": a.track,
        "min_train_docs": a.min_train_docs,
        "min_valid_docs": a.min_valid_docs,
        "skip_trained": a.skip_trained,
        "totals": {
            "rows": len(rows),
            "train_ready": sum(1 for r in rows if r.reason == "train_ready"),
            "needs_build": sum(1 for r in rows if r.reason == "build_data"),
            "below_train_threshold": sum(
                1 for r in rows if r.reason == "train_docs_below_threshold"
            ),
            "below_valid_threshold": sum(
                1 for r in rows if r.reason == "valid_docs_below_threshold"
            ),
        },
        "rows": [asdict(r) for r in rows],
    }

    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Script includes all rows so remote runners can do build + train in one pass.
    _write_shell_script(
        out=a.out_shell,
        rows=rows,
        base_model=a.base_model,
        run_tag_prefix=a.run_tag_prefix,
        hf_repo_prefix=a.hf_repo_prefix,
    )

    print(json.dumps(result["totals"], indent=2))
    print(f"Wrote {a.out_json}")
    print(f"Wrote {a.out_shell}")


if __name__ == "__main__":
    main()
