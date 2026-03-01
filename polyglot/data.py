# polyglot/data.py
"""
Data loading and building for a single language.

Supports multiple sources:
  - mdc: Mozilla Data Collective archives (download + extract)
  - custom: User-provided JSONL files in langs/<code>/sources/

All sources are unified into train.jsonl.zst + valid.jsonl.zst shards
with deduplication via blake2b hashing.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tarfile
import webbrowser
import zipfile
from pathlib import Path
from typing import Iterator

import orjson
import zstandard as zstd
from tqdm import tqdm

from polyglot.config import LangConfig, MdcSource

# Text field names to search for in structured data
TEXT_KEYS = (
    "text",
    "sentence",
    "content",
    "transcript",
    "utterance",
    "prompt",
    "completion",
)


# ---------------------------------------------------------------------------
# Archive extraction (secure)
# ---------------------------------------------------------------------------


def _extract_archive(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                target = (out_dir / member).resolve()
                if not str(target).startswith(str(out_dir.resolve())):
                    raise ValueError(f"Zip path traversal blocked: {member}")
            zf.extractall(out_dir)
        return
    with tarfile.open(archive, "r:*") as tf:
        tf.extractall(out_dir, filter="data")


# ---------------------------------------------------------------------------
# Text extraction from various file formats
# ---------------------------------------------------------------------------


def _iter_txt_docs(p: Path) -> Iterator[str]:
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


def _iter_text_records(root: Path) -> Iterator[str]:
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


def _iter_custom_sources(cfg: LangConfig) -> Iterator[str]:
    """Yield text from any .jsonl files in the sources directory."""
    sources_dir = cfg.sources_dir
    for fp in sorted(sources_dir.glob("*.jsonl")):
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


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _open_dedup_db(cfg: LangConfig) -> sqlite3.Connection:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    db_path = cfg.data_dir / "dedup.sqlite"
    db = sqlite3.connect(str(db_path))
    db.execute("PRAGMA journal_mode=WAL;")
    db.execute("CREATE TABLE IF NOT EXISTS seen (h BLOB PRIMARY KEY)")
    return db


def _seen_insert(db: sqlite3.Connection, h: bytes) -> bool:
    try:
        db.execute("INSERT INTO seen(h) VALUES (?)", (h,))
        return True
    except sqlite3.IntegrityError:
        return False


# ---------------------------------------------------------------------------
# MDC download
# ---------------------------------------------------------------------------


def _dataset_url(did: str) -> str:
    return f"https://datacollective.mozillafoundation.org/datasets/{did}"


def _dir_size_bytes(root: Path) -> int:
    total = 0
    for fp in root.rglob("*"):
        if fp.is_file():
            total += fp.stat().st_size
    return total


def download_mdc(
    cfg: LangConfig, limit: int = 100, max_total_gb: float | None = None
) -> int:
    """Download MDC archives for a language. Returns number of datasets downloaded."""
    from datacollective import save_dataset_to_disk

    raw_dir = cfg.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cap_bytes = int(max_total_gb * (1024**3)) if max_total_gb is not None else None
    done = 0

    for src in cfg.sources_mdc:
        if done >= limit:
            break
        if cap_bytes is not None and _dir_size_bytes(raw_dir) >= cap_bytes:
            print(f"  Size cap reached for {cfg.language}")
            break

        sentinel = raw_dir / src.id / ".done"
        if sentinel.exists():
            continue

        (raw_dir / src.id).mkdir(parents=True, exist_ok=True)
        try:
            archive = save_dataset_to_disk(
                src.id, download_directory=str(raw_dir / src.id)
            )
            sentinel.write_text(str(archive), encoding="utf-8")
            done += 1
        except Exception as e:
            err = str(e)
            (raw_dir / src.id / ".error").write_text(err, encoding="utf-8")
            if "Access denied" in err:
                url = _dataset_url(src.id)
                print(f"  Access denied for {src.id}. Accept terms: {url}")
                try:
                    webbrowser.open_new_tab(url)
                except Exception:
                    pass
    return done


# ---------------------------------------------------------------------------
# Build shards
# ---------------------------------------------------------------------------


def build_shards(cfg: LangConfig, valid_permille: int = 5) -> tuple[int, int]:
    """Build train/valid shards from all sources. Returns (train_n, valid_n)."""
    db = _open_dedup_db(cfg)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    train_cctx = zstd.ZstdCompressor(level=10)
    valid_cctx = zstd.ZstdCompressor(level=10)
    train_n = 0
    valid_n = 0

    with (
        cfg.train_shard.open("wb") as tf,
        cfg.valid_shard.open("wb") as vf,
        train_cctx.stream_writer(tf) as tw,
        valid_cctx.stream_writer(vf) as vw,
    ):
        # Source 1: MDC downloaded archives
        raw_dir = cfg.data_dir / "raw"
        extracted_dir = cfg.data_dir / "extracted"
        for src in tqdm(cfg.sources_mdc, desc=f"mdc:{cfg.language}"):
            sentinel = raw_dir / src.id / ".done"
            if not sentinel.exists():
                continue
            arch = Path(sentinel.read_text(encoding="utf-8").strip())
            exdir = extracted_dir / src.id
            if not exdir.exists():
                _extract_archive(arch, exdir)
            for text in _iter_text_records(exdir):
                t, v = _write_doc(
                    text, cfg.language, src.id, db, tw, vw, valid_permille
                )
                train_n += t
                valid_n += v

        # Source 2: Custom JSONL files in sources/
        for text in _iter_custom_sources(cfg):
            t, v = _write_doc(text, cfg.language, "custom", db, tw, vw, valid_permille)
            train_n += t
            valid_n += v

    db.commit()
    db.close()

    # Write stats
    stats = {"train_docs": train_n, "valid_docs": valid_n}
    cfg.stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return train_n, valid_n


def _write_doc(
    text: str,
    lang: str,
    source_id: str,
    db: sqlite3.Connection,
    tw,
    vw,
    valid_permille: int,
) -> tuple[int, int]:
    """Dedup, hash-split, and write a single document. Returns (train_added, valid_added)."""
    text = text.strip()
    if len(text) < 40:
        return 0, 0
    h = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=16).digest()
    if not _seen_insert(db, h):
        return 0, 0
    row = orjson.dumps({"text": text, "source": source_id, "lang": lang}) + b"\n"
    bucket = h[0]
    if bucket < max(1, int(256 * (valid_permille / 1000))):
        vw.write(row)
        return 0, 1
    else:
        tw.write(row)
        return 1, 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def compute_stats(cfg: LangConfig) -> dict:
    """Count documents in train/valid shards."""

    def count_lines(zst_path: Path) -> int:
        if not zst_path.exists():
            return 0
        dctx = zstd.ZstdDecompressor()
        with zst_path.open("rb") as f, dctx.stream_reader(f) as r:
            return len(r.read().splitlines())

    stats = {
        "train_docs": count_lines(cfg.train_shard),
        "valid_docs": count_lines(cfg.valid_shard),
    }
    cfg.stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
