# ml/models/train.py
"""
LoRA Continued Pre-Training (CPT) on zstd-compressed JSONL shards.

Packs tokens to fixed seq_len for maximum throughput.
Uses PEFT LoRA targeting all attention + MLP projections.

Supports:
  - GPU with optional 4-bit QLoRA (bitsandbytes)
  - TPU via torch_xla (bf16, no quantization)
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
import zstandard as zstd
from peft import LoraConfig, get_peft_model
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def _is_tpu() -> bool:
    """Detect if running on a TPU (torch_xla available and XLA devices present)."""
    try:
        import torch_xla.core.xla_model as xm

        return True
    except ImportError:
        return False


def _tpu_device():
    """Get the XLA device."""
    import torch_xla.core.xla_model as xm

    return xm.xla_device()


def _tpu_core_count() -> int:
    """Get number of TPU cores available."""
    try:
        import torch_xla.core.xla_model as xm

        return xm.xrt_world_size()
    except Exception:
        return 8  # default for TPU v2-8 / v3-8


@dataclass(frozen=True)
class TrainCfg:
    base_model: str
    seq_len: int = 2048
    lr: float = 2e-4
    max_steps: int = 5000
    warmup_steps: int = 200
    grad_accum: int = 16
    per_device_batch_size: int = 1
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


def _load_model_gpu(
    base_model: str,
    load_in_4bit: bool,
    out_dir: Path,
    max_gpu_mem_gb: int | None,
    trust_remote_code: bool,
):
    """Load model for GPU training (optional 4-bit quantization)."""
    if load_in_4bit:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

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
        if max_gpu_mem_gb is not None:
            max_memory = {0: f"{max_gpu_mem_gb}GiB", "cpu": "64GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_cfg,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=str(out_dir / "offload"),
            trust_remote_code=trust_remote_code,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
        )
    return model


def _load_model_tpu(base_model: str, trust_remote_code: bool):
    """Load model for TPU training (bf16, no quantization, no device_map)."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )
    return model


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
    ap.add_argument("--per-device-batch-size", type=int, default=None)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--max-gpu-mem-gb", type=int, default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument(
        "--tpu",
        action="store_true",
        help="Force TPU mode (auto-detected if torch_xla available)",
    )
    a = ap.parse_args()

    use_tpu = a.tpu or _is_tpu()

    # TPU overrides: no 4-bit, larger batch size
    if use_tpu:
        a.load_in_4bit = False
        print(f"TPU mode: bf16, no quantization, cores={_tpu_core_count()}")

    # Default batch size: 1 for GPU, 8 for TPU (more HBM available)
    batch_size = a.per_device_batch_size
    if batch_size is None:
        batch_size = 8 if use_tpu else 1

    # On TPU with larger batch, reduce grad_accum to keep effective batch ~same
    grad_accum = a.grad_accum
    if use_tpu and a.per_device_batch_size is None:
        # Effective batch = cores * batch_size * grad_accum
        # GPU default: 1 * 1 * 16 = 16
        # TPU default: 8 * 8 * 2 = 128 (good for CPT)
        grad_accum = max(1, 2)

    cfg = TrainCfg(
        base_model=a.base_model,
        seq_len=a.seq_len,
        lr=a.lr,
        max_steps=a.max_steps,
        warmup_steps=a.warmup_steps,
        grad_accum=grad_accum,
        per_device_batch_size=batch_size,
        r=a.r,
        alpha=a.alpha,
        dropout=a.dropout,
    )

    tok = AutoTokenizer.from_pretrained(
        cfg.base_model,
        use_fast=True,
        trust_remote_code=a.trust_remote_code,
    )
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    train_ds = ZstJsonlPacked(a.train_zst, tok, cfg.seq_len)
    valid_ds = ZstJsonlPacked(a.valid_zst, tok, cfg.seq_len)

    # Load model
    try:
        if use_tpu:
            model = _load_model_tpu(cfg.base_model, a.trust_remote_code)
        else:
            model = _load_model_gpu(
                cfg.base_model,
                a.load_in_4bit,
                a.out,
                a.max_gpu_mem_gb,
                a.trust_remote_code,
            )
    except ValueError as e:
        msg = str(e)
        if "Unrecognized configuration class" in msg:
            raise ValueError(
                f"Base model '{cfg.base_model}' is not supported by AutoModelForCausalLM in this setup. "
                "Use a supported causal Mistral model such as 'mistralai/Mistral-7B-v0.3' "
                "or pass --trust-remote-code if the model requires custom code."
            ) from e
        raise

    # LoRA config — dropout=0 on TPU avoids non-deterministic ops
    lora = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=0.0 if use_tpu else cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_infer_target_modules(model),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    a.out.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(cfg)
    cfg_dict["tpu"] = use_tpu
    (a.out / "train_cfg.json").write_text(
        json.dumps(cfg_dict, indent=2), encoding="utf-8"
    )

    # Training arguments
    ta_kwargs = {
        "output_dir": str(a.out),
        "max_steps": cfg.max_steps,
        "warmup_steps": cfg.warmup_steps,
        "learning_rate": cfg.lr,
        "per_device_train_batch_size": cfg.per_device_batch_size,
        "per_device_eval_batch_size": cfg.per_device_batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "logging_steps": 25,
        "eval_steps": 500,
        "save_steps": 500,
        "save_total_limit": 2,
        "bf16": True,  # bf16 on both GPU (if supported) and TPU
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_drop_last": use_tpu,  # TPU requires uniform batch shapes
    }

    if use_tpu:
        ta_kwargs["gradient_checkpointing"] = True

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
