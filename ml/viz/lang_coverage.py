#!/usr/bin/env python3
"""
Visualize underrepresented languages in the PolygotAI pipeline.

Produces a multi-panel matplotlib figure showing:
  1. Pipeline status breakdown (donut chart)
  2. Top 30 most underrepresented languages by dataset count (bar chart)
  3. License track distribution across all languages (stacked bar)
  4. Trained vs untrained heatmap grid for data-ready languages

Usage:
    python -m ml.viz.lang_coverage [--out lang_coverage.png]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

try:
    import langcodes

    def _lang_name(code: str) -> str:
        """Resolve ISO 639 code to display name, falling back to the code itself."""
        code = code.strip().split(",")[0].strip()  # handle multi-codes like "ewo, fr"
        try:
            name = langcodes.Language.get(code).display_name()
            if name == code or not name:
                return code
            # Truncate long names
            return name if len(name) <= 20 else name[:18] + "…"
        except Exception:
            return code
except ImportError:

    def _lang_name(code: str) -> str:
        return code.strip()


LANGS_ROOT = Path("langs")
ART = Path("ml/data/artifacts")
BINS_JSON = ART / "manifests" / "bins.json"
COVERAGE_JSON = ART / "manifests" / "coverage.json"

# Color palette
C_TRAINED = "#2ecc71"  # green
C_DATA_READY = "#3498db"  # blue
C_NO_DATA = "#e74c3c"  # red
C_CC0 = "#2ecc71"
C_BY = "#3498db"
C_BY_NC = "#f39c12"
C_OTHER = "#95a5a6"

TRACK_COLORS = {
    "cc0_pd": C_CC0,
    "by": C_BY,
    "by_nc": C_BY_NC,
    "other": C_OTHER,
}
TRACK_LABELS = {
    "cc0_pd": "CC0 / Public Domain",
    "by": "CC-BY",
    "by_nc": "CC-BY-NC",
    "other": "Other / Unknown",
}


def load_data():
    """Load language data from langs/ directory structure (primary) and legacy manifests (fallback)."""
    bins: dict = {}
    coverage: dict = {}

    if BINS_JSON.exists():
        bins = json.loads(BINS_JSON.read_text(encoding="utf-8"))
    if COVERAGE_JSON.exists():
        coverage = json.loads(COVERAGE_JSON.read_text(encoding="utf-8"))

    return bins, coverage


def _load_langs_from_dirs() -> dict[str, dict]:
    """Read all langs/*/lang.yaml and return a dict keyed by language code."""
    try:
        import yaml
    except ImportError:
        return {}

    langs = {}
    for yaml_path in sorted(LANGS_ROOT.glob("*/lang.yaml")):
        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            code = cfg["language"]
            mdc = cfg.get("sources", {}).get("mdc", [])
            custom_count = len(
                list(yaml_path.parent.joinpath("sources").glob("*.jsonl"))
            )

            # Check for data shards
            data_dir = yaml_path.parent / "data"
            has_train = (data_dir / "train.jsonl.zst").exists()
            has_valid = (data_dir / "valid.jsonl.zst").exists()
            stats = {}
            stats_path = data_dir / "stats.json"
            if stats_path.exists():
                try:
                    stats = json.loads(stats_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            # Check for runs
            runs_dir = yaml_path.parent / "runs"
            run_count = 0
            best_ppl = None
            if runs_dir.exists():
                for run_dir in runs_dir.iterdir():
                    if (run_dir / "adapter_model.safetensors").exists():
                        run_count += 1
                    eval_path = run_dir / "eval.json"
                    if eval_path.exists():
                        try:
                            ev = json.loads(eval_path.read_text(encoding="utf-8"))
                            p = ev.get("ppl", float("inf"))
                            if p != float("inf") and (best_ppl is None or p < best_ppl):
                                best_ppl = p
                        except Exception:
                            pass

            # Build track counts from MDC sources
            track_counts: dict[str, int] = {}
            for s in mdc:
                t = s.get("track", "other")
                track_counts[t] = track_counts.get(t, 0) + 1

            langs[code] = {
                "name": cfg.get("name", code),
                "total_datasets": len(mdc) + custom_count,
                "track_counts": track_counts,
                "tracks": list(track_counts.keys()),
                "has_data": has_train and has_valid,
                "is_trained": run_count > 0,
                "train_docs": stats.get("train_docs", 0),
                "best_ppl": best_ppl,
            }
        except Exception:
            continue
    return langs


def build_lang_summary(bins: dict, coverage: dict) -> list[dict]:
    """Build a per-language summary from langs/ dirs, bins, and coverage."""
    cov = coverage.get("coverage", {})

    # Primary: read from langs/ directory structure
    lang_dirs = _load_langs_from_dirs()

    # Merge legacy data for anything not yet in langs/
    _skip = {"smoke", "test"}
    all_codes = (set(lang_dirs.keys()) | set(bins.keys()) | set(cov.keys())) - _skip

    rows = []
    for lang in all_codes:
        # Prefer langs/ dir data if available
        ld = lang_dirs.get(lang)
        if ld:
            status = (
                "trained"
                if ld["is_trained"]
                else ("data_ready" if ld["has_data"] else "no_data")
            )
            rows.append(
                {
                    "lang": lang,
                    "name": ld["name"]
                    if len(ld["name"]) <= 20
                    else ld["name"][:18] + "…",
                    "total_datasets": ld["total_datasets"],
                    "track_counts": ld["track_counts"],
                    "tracks": ld["tracks"],
                    "status": status,
                    "train_docs": ld["train_docs"],
                    "best_ppl": ld["best_ppl"],
                }
            )
            continue

        # Fallback: legacy bins + coverage
        bin_tracks = bins.get(lang, {})
        total_datasets = sum(len(ids) for ids in bin_tracks.values())
        track_list = list(bin_tracks.keys())
        track_counts = {t: len(ids) for t, ids in bin_tracks.items()}

        lang_cov = cov.get(lang, {})
        for t in lang_cov:
            if t not in track_list:
                track_list.append(t)

        has_data = any(
            r.get("has_train_shard") and r.get("has_valid_shard")
            for r in lang_cov.values()
        )
        is_trained = any(bool(r.get("trained_runs")) for r in lang_cov.values())
        train_docs = max(
            (r.get("train_docs") or 0 for r in lang_cov.values()), default=0
        )
        best_ppl = None
        for r in lang_cov.values():
            ev = r.get("best_eval")
            if ev and ev.get("ppl") is not None:
                p = ev["ppl"]
                if p != float("inf") and (best_ppl is None or p < best_ppl):
                    best_ppl = p

        if is_trained:
            status = "trained"
        elif has_data:
            status = "data_ready"
        else:
            status = "no_data"

        rows.append(
            {
                "lang": lang,
                "name": _lang_name(lang),
                "total_datasets": total_datasets,
                "track_counts": track_counts,
                "tracks": track_list,
                "status": status,
                "train_docs": train_docs,
                "best_ppl": best_ppl,
            }
        )

    rows.sort(key=lambda r: r["total_datasets"])
    return rows


def plot(rows: list[dict], out: Path) -> None:
    """Generate the multi-panel visualization."""
    fig = plt.figure(figsize=(20, 14), facecolor="#1a1a2e")
    fig.suptitle(
        "PolygotAI — Underrepresented Language Coverage",
        fontsize=20,
        fontweight="bold",
        color="white",
        y=0.98,
    )

    gs = GridSpec(
        2,
        3,
        figure=fig,
        hspace=0.35,
        wspace=0.30,
        left=0.06,
        right=0.97,
        top=0.92,
        bottom=0.06,
    )

    ax_style = dict(facecolor="#16213e")

    # ── Panel 1: Pipeline status donut ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#16213e")

    status_counts = Counter(r["status"] for r in rows)
    labels = ["No Data Yet", "Data Ready", "Trained"]
    sizes = [
        status_counts.get("no_data", 0),
        status_counts.get("data_ready", 0),
        status_counts.get("trained", 0),
    ]
    colors = [C_NO_DATA, C_DATA_READY, C_TRAINED]
    explode = (0.02, 0.05, 0.05)

    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=None,
        autopct=lambda p: f"{p:.1f}%\n({int(p * sum(sizes) / 100)})",
        colors=colors,
        explode=explode,
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="#1a1a2e", linewidth=2),
        textprops=dict(color="white", fontsize=9),
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_color("white")

    ax1.legend(
        [f"{l} ({s})" for l, s in zip(labels, sizes)],
        loc="lower center",
        fontsize=8,
        facecolor="#16213e",
        edgecolor="#444",
        labelcolor="white",
        framealpha=0.8,
    )
    ax1.set_title(
        "Pipeline Status", fontsize=13, fontweight="bold", color="white", pad=12
    )

    # ── Panel 2: License track distribution ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#16213e")

    track_totals = defaultdict(int)
    for r in rows:
        for t, c in r["track_counts"].items():
            track_totals[t] += c

    track_order = ["cc0_pd", "by", "by_nc", "other"]
    track_vals = [track_totals.get(t, 0) for t in track_order]
    track_labels_display = [TRACK_LABELS.get(t, t) for t in track_order]
    track_colors = [TRACK_COLORS.get(t, "#999") for t in track_order]

    bars = ax2.barh(
        track_labels_display,
        track_vals,
        color=track_colors,
        edgecolor="#1a1a2e",
        height=0.6,
    )
    for bar, val in zip(bars, track_vals):
        ax2.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    ax2.set_xlabel("Total Datasets", color="white", fontsize=10)
    ax2.set_title(
        "License Track Distribution",
        fontsize=13,
        fontweight="bold",
        color="white",
        pad=12,
    )
    ax2.tick_params(colors="white", labelsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_color("#444")
    ax2.spines["left"].set_color("#444")
    ax2.set_xlim(0, max(track_vals) * 1.15)

    # ── Panel 3: Dataset count histogram ────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#16213e")

    ds_counts = [r["total_datasets"] for r in rows]
    max_ds = max(ds_counts)
    bin_edges = list(range(1, min(max_ds + 2, 16)))  # 1..15
    if max_ds >= 15:
        bin_edges.append(max_ds + 1)

    ax3.hist(
        ds_counts,
        bins=bin_edges,
        color="#e74c3c",
        edgecolor="#1a1a2e",
        alpha=0.9,
        rwidth=0.85,
    )
    ax3.set_xlabel("Datasets per Language", color="white", fontsize=10)
    ax3.set_ylabel("Number of Languages", color="white", fontsize=10)
    ax3.set_title(
        "Dataset Count Distribution",
        fontsize=13,
        fontweight="bold",
        color="white",
        pad=12,
    )
    ax3.tick_params(colors="white", labelsize=9)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_color("#444")
    ax3.spines["left"].set_color("#444")

    # Annotate the "1 dataset" bar
    one_ds = sum(1 for c in ds_counts if c == 1)
    ax3.annotate(
        f"{one_ds} languages\nwith only 1 dataset",
        xy=(1, one_ds),
        xytext=(4, one_ds * 0.85),
        fontsize=8,
        color="#ff6b6b",
        arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1.2),
    )

    # ── Panel 4: Bottom-30 most underrepresented languages ──────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor("#16213e")

    # Pick the 30 languages with fewest datasets that are NOT trained
    untrained = [r for r in rows if r["status"] != "trained"]
    bottom_30 = untrained[:30]  # already sorted ascending by total_datasets

    y_labels = [f"{r['name']} ({r['lang']})" for r in bottom_30]
    x_vals = [r["total_datasets"] for r in bottom_30]
    bar_colors = [
        C_DATA_READY if r["status"] == "data_ready" else C_NO_DATA for r in bottom_30
    ]

    bars = ax4.barh(
        range(len(bottom_30)), x_vals, color=bar_colors, edgecolor="#1a1a2e", height=0.7
    )
    ax4.set_yticks(range(len(bottom_30)))
    ax4.set_yticklabels(y_labels, fontsize=7.5, color="white")
    ax4.invert_yaxis()

    for bar, r in zip(bars, bottom_30):
        tracks_str = ", ".join(r["tracks"])
        ax4.text(
            bar.get_width() + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{r['total_datasets']}ds  [{tracks_str}]",
            va="center",
            color="#aaa",
            fontsize=6.5,
        )

    ax4.set_xlabel("Total Datasets", color="white", fontsize=10)
    ax4.set_title(
        "30 Most Underrepresented Languages (untrained)",
        fontsize=13,
        fontweight="bold",
        color="white",
        pad=12,
    )
    # Force integer ticks on x-axis
    max_x = max(x_vals) if x_vals else 1
    ax4.set_xlim(0, max_x + 0.5)
    ax4.set_xticks(range(0, max_x + 2))
    ax4.tick_params(colors="white", labelsize=8)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["bottom"].set_color("#444")
    ax4.spines["left"].set_color("#444")
    ax4.set_xlim(0, max(x_vals) * 1.3 if x_vals else 2)

    legend_patches = [
        mpatches.Patch(color=C_NO_DATA, label="No data shards"),
        mpatches.Patch(color=C_DATA_READY, label="Data ready, not trained"),
    ]
    ax4.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=8,
        facecolor="#16213e",
        edgecolor="#444",
        labelcolor="white",
    )

    # ── Panel 5: Trained languages — perplexity comparison ──────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#16213e")

    trained = [
        r for r in rows if r["status"] == "trained" and r["best_ppl"] is not None
    ]
    trained.sort(key=lambda r: r["best_ppl"])

    if trained:
        t_labels = [f"{r['name']}\n({r['lang']})" for r in trained]
        t_ppls = [r["best_ppl"] for r in trained]
        t_docs = [r["train_docs"] for r in trained]

        bar_colors_t = plt.cm.RdYlGn_r(  # red=high ppl, green=low
            [p / max(t_ppls) for p in t_ppls]
        )
        bars = ax5.bar(
            range(len(trained)),
            t_ppls,
            color=bar_colors_t,
            edgecolor="#1a1a2e",
            width=0.6,
        )
        ax5.set_xticks(range(len(trained)))
        ax5.set_xticklabels(t_labels, fontsize=8, color="white")

        for bar, ppl, r in zip(bars, t_ppls, trained):
            doc_info = f"{r['train_docs']} docs" if r["train_docs"] else ""
            ds_info = f"{r['total_datasets']} ds" if r["total_datasets"] else ""
            sub_label = doc_info or ds_info
            ax5.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8,
                f"PPL {ppl:.0f}" + (f"\n{sub_label}" if sub_label else ""),
                ha="center",
                va="bottom",
                color="white",
                fontsize=7,
            )

        ax5.set_ylabel("Perplexity", color="white", fontsize=10)
        ax5.set_title(
            "Trained Adapters — Perplexity",
            fontsize=13,
            fontweight="bold",
            color="white",
            pad=12,
        )
        ax5.set_ylim(0, max(t_ppls) * 1.25)
    else:
        ax5.text(
            0.5,
            0.5,
            "No trained adapters\nwith eval results",
            ha="center",
            va="center",
            color="#666",
            fontsize=12,
            transform=ax5.transAxes,
        )
        ax5.set_title(
            "Trained Adapters", fontsize=13, fontweight="bold", color="white", pad=12
        )

    ax5.tick_params(colors="white", labelsize=8)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.spines["bottom"].set_color("#444")
    ax5.spines["left"].set_color("#444")

    # ── Summary annotation ──────────────────────────────────────────
    total_langs = len(rows)
    total_ds = sum(r["total_datasets"] for r in rows)
    one_ds_langs = sum(1 for r in rows if r["total_datasets"] == 1)

    fig.text(
        0.5,
        0.01,
        f"Total: {total_langs} languages | {total_ds} datasets | "
        f"{one_ds_langs} languages with only 1 dataset | "
        f"{status_counts.get('trained', 0)} trained | "
        f"{status_counts.get('no_data', 0)} awaiting data",
        ha="center",
        fontsize=10,
        color="#888",
        style="italic",
    )

    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Visualize underrepresented language coverage"
    )
    ap.add_argument("--out", type=Path, default=Path("lang_coverage.png"))
    args = ap.parse_args()

    bins, coverage = load_data()
    rows = build_lang_summary(bins, coverage)
    plot(rows, args.out)


if __name__ == "__main__":
    main()
