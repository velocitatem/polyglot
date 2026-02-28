# ml/models/train.py
"""
LoRA Continued Pre-Training (CPT) on zstd-compressed JSONL shards.

Packs tokens to fixed seq_len for maximum throughput.
Uses PEFT LoRA targeting all attention + MLP projections.
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
import zstandard as zstd
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


@dataclass(frozen=True)
class TrainCfg:
    base_model: str
    seq_len: int = 2048
    lr: float = 2e-4
    max_steps: int = 5000
    warmup_steps: int = 200
    grad_accum: int = 16
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05


class ZstJsonlPacked(IterableDataset):
    """Stream zst-compressed JSONL, tokenize, and pack into fixed-length windows."""

    def __init__(self, zst_path: Path, tok, seq_len: int):
        self.zst_path = zst_path
        self.tok = tok
        self.seq_len = seq_len

    def _iter_texts(self) -> Iterator[str]:
        dctx = zstd.ZstdDecompressor()
        with self.zst_path.open("rb") as f, dctx.stream_reader(f) as r:
            for ln in r.read().splitlines():
                try:
                    obj = json.loads(ln)
                    t = obj.get("text")
                    if isinstance(t, str) and t:
                        yield t
                except Exception:
                    continue

    def __iter__(self):
        eos = self.tok.eos_token_id
        buf: list[int] = []
        for t in self._iter_texts():
            ids = self.tok.encode(t, add_special_tokens=False)
            if eos is not None:
                ids.append(eos)
            if not ids:
                continue
            buf.extend(ids)
            while len(buf) >= self.seq_len + 1:
                x = torch.tensor(buf[: self.seq_len], dtype=torch.long)
                y = torch.tensor(buf[1 : self.seq_len + 1], dtype=torch.long)
                buf = buf[self.seq_len :]
                yield {"input_ids": x, "labels": y}


def _infer_target_modules(model) -> list[str]:
    """Infer LoRA target modules across common decoder architectures."""
    module_names = [name for name, _ in model.named_modules()]

    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    if all(any(name.endswith(t) for name in module_names) for t in preferred):
        return preferred

    candidate_sets = [
        ["c_attn", "c_proj", "c_fc"],
        ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    ]
    for targets in candidate_sets:
        found = [t for t in targets if any(name.endswith(t) for name in module_names)]
        if found:
            return found

    raise ValueError(
        "Could not infer LoRA target modules for this base model. "
        "Pass a model architecture with standard projection names or update mappings."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA CPT training")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train-zst", type=Path, required=True)
    ap.add_argument("--valid-zst", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-gpu-mem-gb", type=int, default=None)
    a = ap.parse_args()

    cfg = TrainCfg(
        base_model=a.base_model,
        seq_len=a.seq_len,
        lr=a.lr,
        max_steps=a.max_steps,
        warmup_steps=a.warmup_steps,
        grad_accum=a.grad_accum,
        r=a.r,
        alpha=a.alpha,
        dropout=a.dropout,
    )

    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    train_ds = ZstJsonlPacked(a.train_zst, tok, cfg.seq_len)
    valid_ds = ZstJsonlPacked(a.valid_zst, tok, cfg.seq_len)

    if a.load_in_4bit:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
            if torch.cuda.is_available()
            else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        max_memory = None
        if a.max_gpu_mem_gb is not None:
            max_memory = {0: f"{a.max_gpu_mem_gb}GiB", "cpu": "64GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=quant_cfg,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=str(a.out / "offload"),
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype="auto")
    lora = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_infer_target_modules(model),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "train_cfg.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )

    ta_kwargs = {
        "output_dir": str(a.out),
        "max_steps": cfg.max_steps,
        "warmup_steps": cfg.warmup_steps,
        "learning_rate": cfg.lr,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": cfg.grad_accum,
        "logging_steps": 25,
        "eval_steps": 500,
        "save_steps": 500,
        "save_total_limit": 2,
        "bf16": torch.cuda.is_available(),
        "report_to": [],
        "remove_unused_columns": False,
    }
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "steps"
    else:
        ta_kwargs["evaluation_strategy"] = "steps"

    args = TrainingArguments(**ta_kwargs)

    Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
    ).train()

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"Adapter saved to {a.out}")


if __name__ == "__main__":
    main()
