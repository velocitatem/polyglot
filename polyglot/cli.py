# polyglot/cli.py
"""
Orchestrator CLI for the PolygotAI multilingual adapter pipeline.

Usage:
    python -m polyglot init [--lang XX | --all]
    python -m polyglot download --lang XX [--limit N]
    python -m polyglot build --lang XX
    python -m polyglot train --lang XX [--base-model ...]
    python -m polyglot eval --lang XX [--run-tag ...]
    python -m polyglot status [--lang XX | --all]
    python -m polyglot publish --lang XX --hf-repo ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from polyglot.config import LangConfig, LANGS_ROOT


def cmd_init(args: argparse.Namespace) -> None:
    """Scaffold language directories from bins.json."""
    from polyglot.scaffold import scaffold_from_bins

    bins_path = Path(args.bins)
    if not bins_path.exists():
        print(f"bins.json not found at {bins_path}")
        print("Run: make mdc-meta && make mdc-bins   (or provide --bins path)")
        sys.exit(1)

    lang = None if args.all else args.lang
    n = scaffold_from_bins(bins_path, lang=lang, overwrite=args.overwrite)
    print(f"Scaffolded {n} language(s) in {LANGS_ROOT}/")


def cmd_download(args: argparse.Namespace) -> None:
    """Download MDC data for a language."""
    from polyglot.data import download_mdc

    cfg = LangConfig.load(args.lang)
    n = download_mdc(cfg, limit=args.limit, max_total_gb=args.max_gb)
    print(f"Downloaded {n} dataset(s) for {cfg.name} ({cfg.language})")


def cmd_build(args: argparse.Namespace) -> None:
    """Build train/valid shards from all sources."""
    from polyglot.data import build_shards

    cfg = LangConfig.load(args.lang)
    train_n, valid_n = build_shards(cfg)
    print(f"{cfg.language}: {train_n} train + {valid_n} valid docs")


def cmd_train(args: argparse.Namespace) -> None:
    """Run LoRA CPT training for a language."""
    import subprocess

    cfg = LangConfig.load(args.lang)
    base_model = args.base_model or cfg.training_param("base_model")
    run_tag = args.run_tag or _auto_run_tag()
    run_dir = cfg.runs_dir / run_tag

    if not cfg.train_shard.exists():
        print(f"No training data. Run: python -m polyglot build --lang {args.lang}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "accelerate",
        "launch",
        "-m",
        "ml.models.train",
        "--base-model",
        base_model,
        "--train-zst",
        str(cfg.train_shard),
        "--valid-zst",
        str(cfg.valid_shard),
        "--out",
        str(run_dir),
        "--seq-len",
        str(cfg.training_param("seq_len")),
        "--lr",
        str(cfg.training_param("lr")),
        "--max-steps",
        str(cfg.training_param("max_steps")),
        "--warmup-steps",
        str(cfg.training_param("warmup_steps")),
        "--grad-accum",
        str(cfg.training_param("grad_accum")),
        "--r",
        str(cfg.training_param("r")),
        "--alpha",
        str(cfg.training_param("alpha")),
        "--dropout",
        str(cfg.training_param("dropout")),
    ]
    if cfg.training_param("load_in_4bit"):
        cmd.append("--load-in-4bit")

    print(f"Training {cfg.language} → {run_dir}")
    subprocess.run(cmd, check=True)


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a trained adapter."""
    import subprocess

    cfg = LangConfig.load(args.lang)
    base_model = args.base_model or cfg.training_param("base_model")
    run_tag = args.run_tag or _latest_run(cfg)
    if not run_tag:
        print(f"No runs found for {args.lang}")
        sys.exit(1)

    run_dir = cfg.runs_dir / run_tag
    cmd = [
        sys.executable,
        "-m",
        "ml.models.eval",
        "--base-model",
        base_model,
        "--adapter",
        str(run_dir),
        "--valid-zst",
        str(cfg.valid_shard),
        "--out",
        str(run_dir / "eval.json"),
    ]
    if cfg.training_param("load_in_4bit"):
        cmd.append("--load-in-4bit")

    print(f"Evaluating {cfg.language} run={run_tag}")
    subprocess.run(cmd, check=True)


def cmd_status(args: argparse.Namespace) -> None:
    """Show pipeline status for one or all languages."""
    if args.lang:
        configs = [LangConfig.load(args.lang)]
    else:
        configs = LangConfig.list_all()

    if not configs:
        print("No languages initialized. Run: python -m polyglot init --all")
        return

    # Header
    print(
        f"{'Lang':<6} {'Name':<22} {'Sources':>7} {'Train':>7} {'Valid':>7} {'Runs':>5} {'Best PPL':>10}"
    )
    print("-" * 75)

    total = len(configs)
    has_data = 0
    has_runs = 0

    for cfg in configs:
        n_sources = len(cfg.sources_mdc) + len(list(cfg.sources_dir.glob("*.jsonl")))
        train_docs = "-"
        valid_docs = "-"
        if cfg.stats_path.exists():
            try:
                st = json.loads(cfg.stats_path.read_text(encoding="utf-8"))
                train_docs = str(st.get("train_docs", 0))
                valid_docs = str(st.get("valid_docs", 0))
                has_data += 1
            except Exception:
                pass
        elif cfg.train_shard.exists():
            train_docs = "?"
            has_data += 1

        runs = (
            sorted(cfg.runs_dir.glob("*/adapter_model.safetensors"))
            if cfg.runs_dir.exists()
            else []
        )
        n_runs = len(runs)
        if n_runs:
            has_runs += 1

        best_ppl = "-"
        for run_dir in cfg.runs_dir.iterdir() if cfg.runs_dir.exists() else []:
            eval_path = run_dir / "eval.json"
            if eval_path.exists():
                try:
                    ev = json.loads(eval_path.read_text(encoding="utf-8"))
                    p = ev.get("ppl", float("inf"))
                    if p != float("inf"):
                        if best_ppl == "-" or p < float(best_ppl):
                            best_ppl = f"{p:.1f}"
                except Exception:
                    pass

        name = cfg.name[:22] if cfg.name else cfg.language
        print(
            f"{cfg.language:<6} {name:<22} {n_sources:>7} {train_docs:>7} {valid_docs:>7} {n_runs:>5} {best_ppl:>10}"
        )

    print("-" * 75)
    print(
        f"Total: {total} languages | {has_data} with data | {has_runs} with trained adapters"
    )


def cmd_publish(args: argparse.Namespace) -> None:
    """Publish adapter to HF Hub."""
    import subprocess

    cfg = LangConfig.load(args.lang)
    run_tag = args.run_tag or _latest_run(cfg)
    if not run_tag:
        print(f"No runs for {args.lang}")
        sys.exit(1)

    run_dir = cfg.runs_dir / run_tag
    cmd = [
        sys.executable,
        "-m",
        "ml.models.publish",
        "--repo",
        args.hf_repo,
        "--adapter",
        str(run_dir),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_run_tag() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_run(cfg: LangConfig) -> str | None:
    if not cfg.runs_dir.exists():
        return None
    runs = sorted(
        [d for d in cfg.runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return runs[0].name if runs else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="polyglot",
        description="PolygotAI — multilingual LoRA adapter pipeline",
    )
    sp = ap.add_subparsers(dest="cmd", required=True)

    # init
    p = sp.add_parser("init", help="Scaffold language directories from MDC bins")
    p.add_argument("--lang", type=str, default=None, help="Single language code")
    p.add_argument("--all", action="store_true", help="Scaffold all languages")
    p.add_argument("--bins", type=str, default="ml/data/artifacts/manifests/bins.json")
    p.add_argument("--overwrite", action="store_true")

    # download
    p = sp.add_parser("download", help="Download MDC data for a language")
    p.add_argument("--lang", type=str, required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max-gb", type=float, default=None)

    # build
    p = sp.add_parser("build", help="Build train/valid shards from all sources")
    p.add_argument("--lang", type=str, required=True)

    # train
    p = sp.add_parser("train", help="LoRA CPT training")
    p.add_argument("--lang", type=str, required=True)
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--run-tag", type=str, default=None)

    # eval
    p = sp.add_parser("eval", help="Evaluate trained adapter")
    p.add_argument("--lang", type=str, required=True)
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--run-tag", type=str, default=None)

    # status
    p = sp.add_parser("status", help="Show pipeline status")
    p.add_argument("--lang", type=str, default=None)

    # publish
    p = sp.add_parser("publish", help="Publish adapter to HF Hub")
    p.add_argument("--lang", type=str, required=True)
    p.add_argument("--hf-repo", type=str, required=True)
    p.add_argument("--run-tag", type=str, default=None)

    a = ap.parse_args()
    {
        "init": cmd_init,
        "download": cmd_download,
        "build": cmd_build,
        "train": cmd_train,
        "eval": cmd_eval,
        "status": cmd_status,
        "publish": cmd_publish,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
