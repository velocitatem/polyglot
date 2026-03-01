# ml/models/eval.py
"""
Evaluate a LoRA adapter by computing held-out perplexity.

Supports GPU (optional 4-bit) and TPU (bf16).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ml.models.train import ZstJsonlPacked


def _is_tpu() -> bool:
    try:
        import torch_xla.core.xla_model as xm

        return True
    except ImportError:
        return False


def _tpu_device():
    import torch_xla.core.xla_model as xm

    return xm.xla_device()


def _get_device(use_tpu: bool) -> torch.device:
    if use_tpu:
        return _tpu_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate LoRA adapter perplexity")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--valid-zst", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--tpu", action="store_true", help="Force TPU mode")
    a = ap.parse_args()

    use_tpu = a.tpu or _is_tpu()

    tok = AutoTokenizer.from_pretrained(a.base_model, use_fast=True)

    if use_tpu:
        # TPU: bf16, no quantization
        model = AutoModelForCausalLM.from_pretrained(
            a.base_model,
            torch_dtype=torch.bfloat16,
        )
    elif a.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            a.base_model,
            quantization_config=quant_cfg,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype="auto")

    model = PeftModel.from_pretrained(model, a.adapter)
    device = _get_device(use_tpu)
    model.eval().to(device)

    ds = ZstJsonlPacked(a.valid_zst, tok, a.seq_len)
    losses: list[float] = []
    for i, batch in enumerate(ds):
        if i >= a.batches:
            break
        batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        losses.append(out.loss.item())

    avg_loss = sum(losses) / max(1, len(losses))
    ppl = math.exp(avg_loss) if losses else float("inf")
    result = {"batches": len(losses), "loss": avg_loss, "ppl": ppl, "tpu": use_tpu}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
