# polyglot/config.py
"""
Language configuration schema and loader.

Each language lives in langs/<code>/lang.yaml with this structure:

    language: fi
    name: Finnish
    family: Uralic
    script: Latin
    sources:
      mdc:
        - id: cmXXXXXX
          track: cc0_pd
      custom: []
    training:
      base_model: mistralai/Ministral-3-14B-Base-2512
      seq_len: 2048
      lr: 0.0002
      max_steps: 5000
      r: 32
      alpha: 64
      dropout: 0.05
      load_in_4bit: true
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


LANGS_ROOT = Path("langs")

TRAINING_DEFAULTS = {
    "base_model": "mistralai/Ministral-3-14B-Base-2512",
    "seq_len": 2048,
    "lr": 2e-4,
    "max_steps": 5000,
    "warmup_steps": 200,
    "grad_accum": 16,
    "r": 32,
    "alpha": 64,
    "dropout": 0.05,
    "load_in_4bit": True,
}


@dataclass
class MdcSource:
    id: str
    track: str


@dataclass
class LangConfig:
    language: str
    name: str
    family: str | None = None
    script: str | None = None
    sources_mdc: list[MdcSource] = field(default_factory=list)
    sources_custom: list[str] = field(default_factory=list)
    training: dict[str, Any] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return LANGS_ROOT / self.language

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def train_shard(self) -> Path:
        return self.data_dir / "train.jsonl.zst"

    @property
    def valid_shard(self) -> Path:
        return self.data_dir / "valid.jsonl.zst"

    @property
    def stats_path(self) -> Path:
        return self.data_dir / "stats.json"

    @property
    def yaml_path(self) -> Path:
        return self.root / "lang.yaml"

    def training_param(self, key: str) -> Any:
        return self.training.get(key, TRAINING_DEFAULTS.get(key))

    def tracks(self) -> set[str]:
        return {s.track for s in self.sources_mdc}

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "language": self.language,
            "name": self.name,
        }
        if self.family:
            d["family"] = self.family
        if self.script:
            d["script"] = self.script
        d["sources"] = {
            "mdc": [{"id": s.id, "track": s.track} for s in self.sources_mdc],
            "custom": self.sources_custom,
        }
        d["training"] = {**TRAINING_DEFAULTS, **self.training}
        return d

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self.yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )

    @classmethod
    def load(cls, lang_code: str) -> LangConfig:
        p = LANGS_ROOT / lang_code / "lang.yaml"
        if not p.exists():
            raise FileNotFoundError(f"No config for language '{lang_code}' at {p}")
        with p.open("r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        sources = d.get("sources", {})
        mdc_list = [
            MdcSource(id=s["id"], track=s["track"]) for s in sources.get("mdc", [])
        ]
        return cls(
            language=d["language"],
            name=d.get("name", lang_code),
            family=d.get("family"),
            script=d.get("script"),
            sources_mdc=mdc_list,
            sources_custom=sources.get("custom", []),
            training=d.get("training", {}),
        )

    @classmethod
    def list_all(cls) -> list[LangConfig]:
        configs = []
        for p in sorted(LANGS_ROOT.glob("*/lang.yaml")):
            try:
                configs.append(cls.load(p.parent.name))
            except Exception:
                continue
        return configs
