# ml/data/etl.py
"""
ETL pipeline for Mozilla Data Collective datasets.

Subcommands:
    meta      - Fetch metadata for each dataset ID
    bins      - Bin datasets by language x license-track
    list      - Print available language/track bins
    download  - Download raw archives (rate-limited by LIMIT)
    build     - Extract, normalize, dedup, shard into train/valid jsonl.zst
    stats     - Count documents in shards
    coverage  - Build language coverage map from bins/data/runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tarfile
import webbrowser
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

import orjson
import zstandard as zstd
from tqdm import tqdm

from datacollective import get_dataset_details, save_dataset_to_disk


ART = Path("ml/data/artifacts")
MAN = ART / "manifests"
RAW = ART / "raw"
EXT = ART / "extracted"
NORM = ART / "normalized"
DEDUP = ART / "dedup"

DATASET_URL_RE = re.compile(r"/datasets/([a-zA-Z0-9]+)\b")
TEXT_KEYS = (
    "text",
    "sentence",
    "content",
    "transcript",
    "utterance",
    "prompt",
    "completion",
)


LicenseTrack = Literal["cc0_pd", "by", "by_nc", "other"]


@dataclass(frozen=True)
class DatasetRef:
    id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    for p in (MAN, RAW, EXT, NORM, DEDUP):
        p.mkdir(parents=True, exist_ok=True)


def _extract_id(line: str) -> str | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    m = DATASET_URL_RE.search(s)
    return m.group(1) if m else s


def _read_ids(datasets_txt: Path) -> list[str]:
    ids = [
        _extract_id(x) for x in datasets_txt.read_text(encoding="utf-8").splitlines()
    ]
    return sorted({x for x in ids if x})


def _license_track(lic: str | None) -> LicenseTrack:
    u = (lic or "").upper()
    if u in {"CC0-1.0", "UNLICENSE"}:
        return "cc0_pd"
    if "BY-NC" in u:
        return "by_nc"
    if "CC-BY" in u:
        return "by"
    return "other"


def _norm_lang(locale: str) -> str:
    return locale.replace("_", "-").split("-")[0].lower()


def _dataset_url(did: str) -> str:
    return f"https://datacollective.mozillafoundation.org/datasets/{did}"


def _zst_write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_cctx = zstd.ZstdCompressor(level=10)
    valid_cctx = zstd.ZstdCompressor(level=10)
    with path.open("wb") as f, cctx.stream_writer(f) as w:
        for r in rows:
            w.write(orjson.dumps(r) + b"\n")


def _open_dedup_db(lang: str, track: str) -> sqlite3.Connection:
    DEDUP.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DEDUP / f"{lang}__{track}.sqlite"))
    db.execute("PRAGMA journal_mode=WAL;")
    db.execute("CREATE TABLE IF NOT EXISTS seen (h BLOB PRIMARY KEY)")
    return db


def _seen_insert(db: sqlite3.Connection, h: bytes) -> bool:
    """Insert hash into dedup table. Returns True if new (not a duplicate)."""
    try:
        db.execute("INSERT INTO seen(h) VALUES (?)", (h,))
        return True
    except sqlite3.IntegrityError:
        return False


# ---------------------------------------------------------------------------
# Subcommand: meta
# ---------------------------------------------------------------------------


def meta_cmd(datasets_txt: Path, out: Path) -> None:
    """Fetch metadata for every dataset ID via the MDC API."""
    _ensure_dirs()
    ids = _read_ids(datasets_txt)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        for did in tqdm(ids, desc="mdc:meta"):
            try:
                f.write(orjson.dumps(get_dataset_details(did)) + b"\n")
            except Exception as e:
                f.write(orjson.dumps({"id": did, "error": str(e)}) + b"\n")


# ---------------------------------------------------------------------------
# Subcommand: bins
# ---------------------------------------------------------------------------


def bins_cmd(meta: Path, out: Path) -> None:
    """Bin datasets into language x license-track from metadata JSONL."""
    _ensure_dirs()
    bins: dict[str, dict[str, list[str]]] = {}
    for line in meta.read_bytes().splitlines():
        obj = orjson.loads(line)
        if "error" in obj:
            continue
        did = obj.get("id") or obj.get("datasetId")
        lic = obj.get("license")
        loc = obj.get("locale")
        locales = loc if isinstance(loc, list) else [loc]
        locales = [x for x in locales if isinstance(x, str) and x.strip()]
        if not did or not locales:
            continue
        track = _license_track(lic)
        for locale in locales:
            lang = _norm_lang(locale)
            bins.setdefault(lang, {}).setdefault(track, []).append(did)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(bins, option=orjson.OPT_INDENT_2))


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------


def list_cmd(bins: Path) -> None:
    """Print available language/track bins."""
    b = json.loads(bins.read_text(encoding="utf-8"))
    for lang in sorted(b.keys()):
        tracks = ",".join(sorted(b[lang].keys()))
        print(f"{lang}\t{tracks}")


# ---------------------------------------------------------------------------
# Subcommand: download
# ---------------------------------------------------------------------------


def download_cmd(bins: Path, limit: int, lang: str | None, track: str | None) -> None:
    """Download raw archives from MDC, respecting a daily cap via LIMIT."""
    _ensure_dirs()
    b = json.loads(bins.read_text(encoding="utf-8"))
    todo: list[str] = []
    for l, tracks in b.items():
        if lang and l != lang:
            continue
        for t, ids in tracks.items():
            if track and t != track:
                continue
            todo.extend(ids)

    done = 0
    for did in todo:
        if done >= limit:
            break
        sentinel = RAW / did / ".done"
        if sentinel.exists():
            continue
        (RAW / did).mkdir(parents=True, exist_ok=True)
        try:
            archive = save_dataset_to_disk(did, download_directory=str(RAW / did))
            sentinel.write_text(str(archive), encoding="utf-8")
            done += 1
        except Exception as e:
            err = str(e)
            (RAW / did / ".error").write_text(err, encoding="utf-8")
            if "Access denied" in err:
                url = _dataset_url(did)
                print(f"Access denied for {did}. Open and accept terms: {url}")
                try:
                    webbrowser.open_new_tab(url)
                except Exception:
                    pass
    print(f"Downloaded {done} datasets (limit={limit})")


# ---------------------------------------------------------------------------
# Subcommand: build
# ---------------------------------------------------------------------------


def _extract_archive(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        return
    with tarfile.open(archive, "r:*") as tf:
        tf.extractall(out_dir)


def _iter_txt_docs(p: Path) -> Iterator[str]:
    """Yield documents separated by blank lines from a plain text file."""
    buf: list[str] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            if buf:
                yield "\n".join(buf).strip()
                buf = []
        else:
            buf.append(line)
    if buf:
        yield "\n".join(buf).strip()


def _guess_text_field(keys: list[str]) -> str:
    lk = [k.lower() for k in keys]
    for k in TEXT_KEYS:
        if k in lk:
            return keys[lk.index(k)]
    return keys[0] if keys else "text"


def _iter_text_records(root: Path) -> Iterator[str]:
    """Walk extracted files and yield text strings."""
    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        suf = fp.suffix.lower()
        if suf in {".txt", ".md"}:
            yield from _iter_txt_docs(fp)
        elif suf == ".jsonl":
            for ln in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                for k in TEXT_KEYS:
                    v = obj.get(k)
                    if isinstance(v, str) and len(v) >= 20:
                        yield v
                        break
        elif suf == ".json":
            try:
                obj = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            if isinstance(obj, list):
                for it in obj:
                    if isinstance(it, dict):
                        for k in TEXT_KEYS:
                            v = it.get(k)
                            if isinstance(v, str) and len(v) >= 20:
                                yield v
                                break
            elif isinstance(obj, dict):
                k = _guess_text_field(list(obj.keys()))
                v = obj.get(k)
                if isinstance(v, str) and len(v) >= 20:
                    yield v


def build_cmd(bins: Path, lang: str, track: str, valid_permille: int = 5) -> None:
    """Extract, normalize, dedup, and shard into train/valid jsonl.zst."""
    _ensure_dirs()
    b = json.loads(bins.read_text(encoding="utf-8"))
    ids: list[str] = b.get(lang, {}).get(track, [])
    if not ids:
        raise SystemExit(f"Empty bin: lang={lang} track={track}")

    db = _open_dedup_db(lang, track)
    train_out = NORM / f"lang={lang}" / f"track={track}" / "train.jsonl.zst"
    valid_out = NORM / f"lang={lang}" / f"track={track}" / "valid.jsonl.zst"

    train_out.parent.mkdir(parents=True, exist_ok=True)
    valid_out.parent.mkdir(parents=True, exist_ok=True)
    train_cctx = zstd.ZstdCompressor(level=10)
    valid_cctx = zstd.ZstdCompressor(level=10)
    train_n = 0
    valid_n = 0

    with (
        train_out.open("wb") as tf,
        valid_out.open("wb") as vf,
        train_cctx.stream_writer(tf) as tw,
        valid_cctx.stream_writer(vf) as vw,
    ):
        for did in tqdm(ids, desc=f"build:{lang}:{track}"):
            sentinel = RAW / did / ".done"
            if not sentinel.exists():
                continue
            arch = Path(sentinel.read_text(encoding="utf-8").strip())
            exdir = EXT / did
            if not exdir.exists():
                _extract_archive(arch, exdir)
            for text in _iter_text_records(exdir):
                text = text.strip()
                if len(text) < 40:
                    continue
                h = hashlib.blake2b(
                    text.encode("utf-8", errors="ignore"), digest_size=16
                ).digest()
                if not _seen_insert(db, h):
                    continue

                row = {
                    "text": text,
                    "dataset_id": did,
                    "lang": lang,
                    "track": track,
                }

                # Deterministic split via first byte of hash
                bucket = h[0]
                is_valid = bucket < max(1, int(256 * (valid_permille / 1000)))
                if is_valid:
                    vw.write(orjson.dumps(row) + b"\n")
                    valid_n += 1
                else:
                    tw.write(orjson.dumps(row) + b"\n")
                    train_n += 1

    db.commit()
    db.close()
    print(f"Wrote {train_out} ({train_n}) and {valid_out} ({valid_n})")


# ---------------------------------------------------------------------------
# Subcommand: stats
# ---------------------------------------------------------------------------


def stats_cmd(lang: str, track: str) -> None:
    """Count documents in train/valid shards."""
    p = NORM / f"lang={lang}" / f"track={track}"
    train = p / "train.jsonl.zst"
    valid = p / "valid.jsonl.zst"
    out = p / "stats.json"

    def count_lines(zst_path: Path) -> int:
        dctx = zstd.ZstdDecompressor()
        n = 0
        with zst_path.open("rb") as f, dctx.stream_reader(f) as r:
            for _ in r.read().splitlines():
                n += 1
        return n

    result = {
        "train_docs": count_lines(train),
        "valid_docs": count_lines(valid),
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Subcommand: coverage
# ---------------------------------------------------------------------------


def _parse_eq_part(part: str, key: str) -> str | None:
    prefix = f"{key}="
    if not part.startswith(prefix):
        return None
    return part[len(prefix) :]


def coverage_cmd(
    out: Path,
    bins: Path | None = None,
    base_model: str | None = None,
    track_filter: str | None = None,
) -> None:
    """Build a language/track coverage map from bins, normalized data, and runs."""
    _ensure_dirs()

    coverage: dict[str, dict[str, dict]] = {}

    # Seed from bins, when available.
    if bins and bins.exists():
        b = json.loads(bins.read_text(encoding="utf-8"))
        for lang, tracks in b.items():
            for track, ids in tracks.items():
                if track_filter and track != track_filter:
                    continue
                coverage.setdefault(lang, {}).setdefault(track, {})["dataset_count"] = (
                    len(ids)
                )

    # Mark normalized shards + optional stats.
    for train_path in NORM.glob("lang=*/track=*/train.jsonl.zst"):
        track_part = train_path.parent.name
        lang_part = train_path.parent.parent.name
        lang = _parse_eq_part(lang_part, "lang")
        track = _parse_eq_part(track_part, "track")
        if not lang or not track:
            continue
        if track_filter and track != track_filter:
            continue

        row = coverage.setdefault(lang, {}).setdefault(track, {})
        row["has_train_shard"] = True
        row.setdefault(
            "has_valid_shard", (train_path.parent / "valid.jsonl.zst").exists()
        )

        stats_path = train_path.parent / "stats.json"
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                row["train_docs"] = int(stats.get("train_docs", 0))
                row["valid_docs"] = int(stats.get("valid_docs", 0))
            except Exception:
                pass

    for valid_path in NORM.glob("lang=*/track=*/valid.jsonl.zst"):
        track_part = valid_path.parent.name
        lang_part = valid_path.parent.parent.name
        lang = _parse_eq_part(lang_part, "lang")
        track = _parse_eq_part(track_part, "track")
        if not lang or not track:
            continue
        if track_filter and track != track_filter:
            continue
        row = coverage.setdefault(lang, {}).setdefault(track, {})
        row["has_valid_shard"] = True

    # Discover training/eval artifacts by base/lang/track/run.
    runs_root = ART / "runs"
    for run_dir in runs_root.glob("base=*/lang=*/track=*/*"):
        if not run_dir.is_dir():
            continue
        track_part = run_dir.parent.name
        lang_part = run_dir.parent.parent.name
        base_part = run_dir.parent.parent.parent.name
        track = _parse_eq_part(track_part, "track")
        lang = _parse_eq_part(lang_part, "lang")
        base = _parse_eq_part(base_part, "base")
        if not lang or not track or not base:
            continue
        if track_filter and track != track_filter:
            continue
        if base_model and base != base_model.replace("/", "_"):
            continue

        row = coverage.setdefault(lang, {}).setdefault(track, {})
        run_tag = run_dir.name

        if (run_dir / "adapter_model.safetensors").exists():
            trained_runs = row.setdefault("trained_runs", {})
            trained_runs.setdefault(base, []).append(run_tag)

        eval_path = run_dir / "eval.json"
        if eval_path.exists():
            try:
                ev = json.loads(eval_path.read_text(encoding="utf-8"))
                result = {
                    "base": base,
                    "run": run_tag,
                    "loss": float(ev.get("loss")),
                    "ppl": float(ev.get("ppl")),
                }
                existing = row.get("best_eval")
                if not existing or result["loss"] < existing["loss"]:
                    row["best_eval"] = result
            except Exception:
                pass

    # Normalize row defaults.
    for tracks in coverage.values():
        for row in tracks.values():
            row.setdefault("dataset_count", None)
            row.setdefault("has_train_shard", False)
            row.setdefault("has_valid_shard", False)
            row.setdefault("train_docs", None)
            row.setdefault("valid_docs", None)
            row.setdefault("trained_runs", {})

    # Build queue of language/track bins that are data-ready but not trained for base.
    queue: list[dict] = []
    data_queue: list[dict] = []
    for lang, tracks in sorted(coverage.items()):
        for track, row in sorted(tracks.items()):
            if row["dataset_count"] is not None and not (
                row["has_train_shard"] and row["has_valid_shard"]
            ):
                data_queue.append(
                    {
                        "lang": lang,
                        "track": track,
                        "dataset_count": row["dataset_count"],
                    }
                )
                continue
            if not (row["has_train_shard"] and row["has_valid_shard"]):
                continue
            if row["dataset_count"] is None:
                continue
            if base_model:
                base_key = base_model.replace("/", "_")
                if row["trained_runs"].get(base_key):
                    continue
            elif row["trained_runs"]:
                continue
            queue.append(
                {
                    "lang": lang,
                    "track": track,
                    "dataset_count": row["dataset_count"],
                    "train_docs": row["train_docs"],
                }
            )

    result = {
        "base_model_filter": base_model,
        "track_filter": track_filter,
        "totals": {
            "languages": len(coverage),
            "language_tracks": sum(len(x) for x in coverage.values()),
            "data_ready": sum(
                1
                for tracks in coverage.values()
                for row in tracks.values()
                if row["has_train_shard"] and row["has_valid_shard"]
            ),
            "trained": sum(
                1
                for tracks in coverage.values()
                for row in tracks.values()
                if row["trained_runs"]
            ),
        },
        "data_queue": data_queue,
        "train_queue": queue,
        "coverage": coverage,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["totals"], indent=2))
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PolygotAI ETL pipeline for Mozilla Data Collective datasets"
    )
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("meta", help="Fetch dataset metadata from MDC API")
    p.add_argument("--datasets-txt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    p = sp.add_parser("bins", help="Bin datasets by language x license-track")
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    p = sp.add_parser("list", help="List available bins")
    p.add_argument("--bins", type=Path, required=True)

    p = sp.add_parser("download", help="Download raw archives from MDC")
    p.add_argument("--bins", type=Path, required=True)
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--lang", type=str, default=None)
    p.add_argument("--track", type=str, default=None)

    p = sp.add_parser("build", help="Normalize and shard into train/valid")
    p.add_argument("--bins", type=Path, required=True)
    p.add_argument("--lang", type=str, required=True)
    p.add_argument("--track", type=str, required=True)
    p.add_argument("--valid-permille", type=int, default=5)

    p = sp.add_parser("stats", help="Count documents in shards")
    p.add_argument("--lang", type=str, required=True)
    p.add_argument("--track", type=str, required=True)

    p = sp.add_parser("coverage", help="Build language coverage map")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bins", type=Path, default=None)
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--track", type=str, default=None)

    a = ap.parse_args()
    if a.cmd == "meta":
        meta_cmd(a.datasets_txt, a.out)
    elif a.cmd == "bins":
        bins_cmd(a.meta, a.out)
    elif a.cmd == "list":
        list_cmd(a.bins)
    elif a.cmd == "download":
        download_cmd(a.bins, a.limit, a.lang, a.track)
    elif a.cmd == "build":
        build_cmd(a.bins, a.lang, a.track, a.valid_permille)
    elif a.cmd == "stats":
        stats_cmd(a.lang, a.track)
    elif a.cmd == "coverage":
        coverage_cmd(a.out, a.bins, a.base_model, a.track)


if __name__ == "__main__":
    main()
