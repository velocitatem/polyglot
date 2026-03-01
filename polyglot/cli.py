# polyglot/cli.py
"""
Orchestrator CLI for the PolygotAI multilingual adapter pipeline.

Usage:
    python -m polyglot init [--lang XX | --all]
    python -m polyglot download --lang XX [--limit N]
    python -m polyglot build --lang XX
    python -m polyglot train --lang XX [--base-model ...]
    python -m polyglot eval --lang XX [--run-tag ...]
    python -m polyglot migrate [--overwrite]
    python -m polyglot status [--lang XX | --all]
    python -m polyglot publish --lang XX --hf-repo ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from polyglot.config import LangConfig, LANGS_ROOT


UNSUPPORTED_BASE_MODELS = {
    "mistralai/Ministral-3-14B-Base-2512": "mistralai/Mistral-7B-v0.3",
}


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
    raw_done = len(list((cfg.data_dir / "raw").glob("*/.done")))
    custom_jsonl = len(list(cfg.sources_dir.glob("*.jsonl")))
    if raw_done == 0 and custom_jsonl == 0 and cfg.sources_mdc:
        print(
            f"No local MDC downloads found for {cfg.language}. "
            f"Run: python -m polyglot download --lang {args.lang}"
        )
        sys.exit(1)

    train_n, valid_n = build_shards(cfg)
    print(f"{cfg.language}: {train_n} train + {valid_n} valid docs")
    if train_n == 0 and valid_n == 0:
        errors = list((cfg.data_dir / "raw").glob("*/.error"))
        if errors:
            print(
                f"Found {len(errors)} download error file(s) under {cfg.data_dir / 'raw'}; "
                "check access/terms for those datasets."
            )
        else:
            print(
                "No valid text records found. Add custom JSONL in "
                f"{cfg.sources_dir} or re-run download."
            )


def cmd_train(args: argparse.Namespace) -> None:
    """Run LoRA CPT training for a language."""
    import subprocess
    from polyglot.config import TRAINING_DEFAULTS_TPU
    from polyglot.data import compute_stats

    cfg = LangConfig.load(args.lang)
    use_tpu = getattr(args, "tpu", False)
    if use_tpu:
        _ensure_tpu_runtime()

    # On TPU, override training params with TPU defaults (unless user set them in lang.yaml)
    def _param(key: str):
        """Get param: lang.yaml override > TPU defaults (if --tpu) > normal defaults."""
        # If explicitly set in lang.yaml training section, use it
        if key in cfg.training:
            return cfg.training[key]
        if use_tpu:
            return TRAINING_DEFAULTS_TPU.get(key, cfg.training_param(key))
        return cfg.training_param(key)

    base_model = _resolve_base_model(
        args.base_model or _param("base_model"), args.base_model is not None
    )
    trust_remote_code = bool(_param("trust_remote_code"))
    run_tag = args.run_tag or _auto_run_tag()
    run_dir = cfg.runs_dir / run_tag

    if not cfg.train_shard.exists() or not cfg.valid_shard.exists():
        print(
            "Missing train/valid shards. "
            f"Run: python -m polyglot download --lang {args.lang} && "
            f"python -m polyglot build --lang {args.lang}"
        )
        sys.exit(1)

    stats = compute_stats(cfg)
    if stats.get("train_docs", 0) <= 0:
        print(
            f"No training docs for {args.lang}. "
            f"Run: python -m polyglot download --lang {args.lang} then build."
        )
        sys.exit(1)
    if stats.get("valid_docs", 0) <= 0:
        print(
            f"No validation docs for {args.lang}. "
            "Training would be unstable; add more data and rebuild."
        )
        sys.exit(1)

    accel = _accelerate_executable()
    cmd = [
        accel,
        "launch",
    ]
    if use_tpu:
        cmd.extend(
            [
                "--tpu",
                "--num_processes",
                str(_tpu_core_count()),
                "--mixed_precision",
                "bf16",
                "--main_training_function",
                "main",
            ]
        )
    cmd.extend(
        [
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
            str(_param("seq_len")),
            "--lr",
            str(_param("lr")),
            "--max-steps",
            str(_param("max_steps")),
            "--warmup-steps",
            str(_param("warmup_steps")),
            "--grad-accum",
            str(_param("grad_accum")),
            "--r",
            str(_param("r")),
            "--alpha",
            str(_param("alpha")),
            "--dropout",
            str(_param("dropout")),
        ]
    )

    batch_size = _param("per_device_batch_size")
    if batch_size is not None:
        cmd.extend(["--per-device-batch-size", str(batch_size)])

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if use_tpu:
        cmd.append("--tpu")
    elif _param("load_in_4bit"):
        cmd.append("--load-in-4bit")

    print(f"Training {cfg.language} → {run_dir}" + (" [TPU]" if use_tpu else ""))
    subprocess.run(cmd, check=True, env=_child_env())


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a trained adapter."""
    import subprocess
    from polyglot.data import compute_stats

    cfg = LangConfig.load(args.lang)
    use_tpu = getattr(args, "tpu", False)
    if use_tpu:
        _ensure_tpu_runtime()
    base_model = _resolve_base_model(
        args.base_model or cfg.training_param("base_model"), args.base_model is not None
    )
    trust_remote_code = bool(cfg.training_param("trust_remote_code"))
    run_tag = args.run_tag or _latest_run(cfg)
    if not run_tag:
        print(f"No runs found for {args.lang}")
        sys.exit(1)

    run_dir = cfg.runs_dir / run_tag
    stats = compute_stats(cfg)
    if stats.get("valid_docs", 0) <= 0:
        print(
            f"No validation docs for {args.lang}. "
            f"Run: python -m polyglot download --lang {args.lang} && "
            f"python -m polyglot build --lang {args.lang}"
        )
        sys.exit(1)

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
    if use_tpu:
        cmd.append("--tpu")
    elif cfg.training_param("load_in_4bit"):
        cmd.append("--load-in-4bit")

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    print(f"Evaluating {cfg.language} run={run_tag}" + (" [TPU]" if use_tpu else ""))
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


def cmd_migrate(args: argparse.Namespace) -> None:
    """Migrate legacy normalized shards and runs into langs/<code>/ structure."""
    from polyglot.data import compute_stats

    legacy_norm_root = Path(args.legacy_normalized)
    legacy_runs_root = Path(args.legacy_runs)

    migrated_shards = 0
    migrated_runs = 0

    if legacy_norm_root.exists():
        for track_dir in sorted(legacy_norm_root.glob("lang=*/track=*")):
            lang_dir = track_dir.parent.name
            if not lang_dir.startswith("lang="):
                continue
            lang = lang_dir.split("=", 1)[1]
            try:
                cfg = LangConfig.load(lang)
            except Exception:
                continue

            src_train = track_dir / "train.jsonl.zst"
            src_valid = track_dir / "valid.jsonl.zst"
            if not src_train.exists() or not src_valid.exists():
                continue

            cfg.data_dir.mkdir(parents=True, exist_ok=True)
            dst_train = cfg.train_shard
            dst_valid = cfg.valid_shard

            if not args.overwrite and dst_train.exists() and dst_valid.exists():
                continue

            shutil.copy2(src_train, dst_train)
            shutil.copy2(src_valid, dst_valid)
            compute_stats(cfg)
            migrated_shards += 1

    if legacy_runs_root.exists():
        for run_dir in sorted(legacy_runs_root.glob("base=*/lang=*/track=*/*")):
            if not run_dir.is_dir():
                continue
            parts = run_dir.parts
            try:
                base_raw = next(p for p in parts if p.startswith("base="))
                lang_raw = next(p for p in parts if p.startswith("lang="))
                track_raw = next(p for p in parts if p.startswith("track="))
            except StopIteration:
                continue

            lang = lang_raw.split("=", 1)[1]
            base = base_raw.split("=", 1)[1]
            track = track_raw.split("=", 1)[1]

            try:
                cfg = LangConfig.load(lang)
            except Exception:
                continue

            safe_base = base.replace("/", "_")
            safe_track = track.replace("/", "_")
            target_name = f"legacy-{safe_base}-{safe_track}-{run_dir.name}"
            target_dir = cfg.runs_dir / target_name

            if target_dir.exists() and not args.overwrite:
                continue

            target_dir.parent.mkdir(parents=True, exist_ok=True)
            if target_dir.exists() and args.overwrite:
                shutil.rmtree(target_dir)
            shutil.copytree(run_dir, target_dir)
            migrated_runs += 1

    print(
        f"Migration complete: {migrated_shards} language shard set(s), "
        f"{migrated_runs} run(s)."
    )


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


def _accelerate_executable() -> str:
    """Prefer accelerate in active venv; fallback to PATH."""
    venv_accelerate = Path(sys.executable).with_name("accelerate")
    if venv_accelerate.exists():
        return str(venv_accelerate)
    return "accelerate"


def _tpu_core_count() -> int:
    try:
        import torch_xla.core.xla_model as xm

        n = xm.xrt_world_size()
        return n if isinstance(n, int) and n > 0 else 8
    except Exception:
        return 8


def _ensure_tpu_runtime() -> None:
    """Fail fast with actionable TPU runtime diagnostics."""
    try:
        import torch_xla  # noqa: F401
    except Exception as e:
        msg = str(e)
        print("TPU runtime check failed: could not import torch_xla.")
        print(f"Error: {msg}")
        if "GLIBC_" in msg:
            print(
                "Detected a GLIBC mismatch between this VM and installed torch_xla wheel."
            )
            print("Fix on TPU VM:")
            print("  1) make deps")
            print("  2) make deps-tpu")
            print(
                "  3) If it still fails, install a torch/torch_xla pair built for your VM image"
            )
        sys.exit(1)


def _child_env() -> dict[str, str]:
    """Child env with repo root available on PYTHONPATH for launchers."""
    env = os.environ.copy()
    cwd = str(Path.cwd())
    py_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = cwd if not py_path else f"{cwd}:{py_path}"
    return env


def _resolve_base_model(base_model: str, is_explicit: bool) -> str:
    """Swap known incompatible defaults unless user explicitly requested them."""
    fallback = UNSUPPORTED_BASE_MODELS.get(base_model)
    if fallback and not is_explicit:
        print(
            "Configured base model is not compatible with current training loader. "
            f"Using fallback: {fallback}"
        )
        return fallback
    return base_model


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
    p.add_argument("--tpu", action="store_true", help="Use TPU (bf16, no quantization)")

    # eval
    p = sp.add_parser("eval", help="Evaluate trained adapter")
    p.add_argument("--lang", type=str, required=True)
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--run-tag", type=str, default=None)
    p.add_argument("--tpu", action="store_true", help="Use TPU for evaluation")

    # status
    p = sp.add_parser("status", help="Show pipeline status")
    p.add_argument("--lang", type=str, default=None)

    # migrate
    p = sp.add_parser("migrate", help="Migrate legacy artifacts into langs/<code>")
    p.add_argument(
        "--legacy-normalized",
        type=str,
        default="ml/data/artifacts/normalized",
    )
    p.add_argument("--legacy-runs", type=str, default="ml/data/artifacts/runs")
    p.add_argument("--overwrite", action="store_true")

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
        "migrate": cmd_migrate,
        "status": cmd_status,
        "publish": cmd_publish,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
