# ml/models/eval.py
"""
Evaluate a LoRA adapter by computing held-out perplexity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ml.models.train import ZstJsonlPacked


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate LoRA adapter perplexity")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--valid-zst", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--load-in-4bit", action="store_true")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base_model, use_fast=True)
    if a.load_in_4bit:
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    result = {"batches": len(losses), "loss": avg_loss, "ppl": ppl}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
