"""Kaggle GPU training kernel for gliner2-base + SCV LoRA adapter.

Attach this file + the training dataset (train.jsonl / eval.jsonl) to a Kaggle
kernel via kernel-metadata.json or the ij717_logic.push_kernel() helper.

Requires the kernel to be pushed with:
  kaggle kernels push --accelerator NvidiaTeslaT4

The T4 (sm_75) is fully supported by Kaggle's stock torch — do NOT pin/replace
torch (the P100 Pascal sm_60 isn't supported and causes CUDA arch errors).
"""
import os
import sys
import subprocess
import glob

# ── Config (replace with EPOCHS/BATCH literals when push_kernel injects them) ──
EPOCHS = int(os.environ.get("EPOCHS", "4"))
BATCH = int(os.environ.get("BATCH", "8"))
BASE_MODEL = "fastino/gliner2-base-v1"

# ── Install — no-deps preserves Kaggle's GPU-matched torch ────────────────────
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-deps", "gliner2==1.3.1", "gliner"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "peft", "accelerate", "seqeval", "onnxruntime", "sentencepiece"])

import torch
print("CUDA:", torch.cuda.is_available(), torch.__version__,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
# Fail fast with a real kernel op (allocation alone passes on broken Pascal).
if torch.cuda.is_available():
    _ = (torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")).sum().item()
    print("CUDA matmul ok")

# ── Locate dataset (mount path varies: /kaggle/input/<slug>/ or nested) ───────
_tr = glob.glob("/kaggle/input/**/train.jsonl", recursive=True)
_ev = glob.glob("/kaggle/input/**/eval.jsonl", recursive=True)
train = _tr[0] if _tr else "/kaggle/input/train.jsonl"
ev = _ev[0] if _ev else ""
print(f"train: {train} (exists={os.path.exists(train)}) | eval: {ev}")

# ── Train ─────────────────────────────────────────────────────────────────────
from gliner2.training.trainer import TrainingConfig, train_gliner2

cfg = TrainingConfig(
    output_dir="/kaggle/working/out",
    experiment_name="scv-gliner2-kaggle",
    num_epochs=EPOCHS,
    batch_size=BATCH,
    eval_batch_size=16,
    gradient_accumulation_steps=2,
    encoder_lr=1e-5,
    task_lr=5e-4,
    warmup_ratio=0.1,
    fp16=False,
    bf16=torch.cuda.is_bf16_supported(),
    use_lora=True,
    lora_r=16,
    lora_alpha=32.0,
    lora_dropout=0.05,
    lora_target_modules=["encoder", "span_rep", "classifier", "count_embed", "count_pred"],
    save_adapter_only=True,
    eval_strategy="epoch" if os.path.exists(ev) else "no",
    save_total_limit=2,
    save_best=os.path.exists(ev),
    metric_for_best="loss",
    greater_is_better=False,
    logging_steps=10,
    seed=42,
    num_workers=2,
)

tk = {"model_path": BASE_MODEL, "train_data": train, "output_dir": "/kaggle/working/out"}
if os.path.exists(ev):
    tk["eval_data"] = ev
kw = {k: v for k, v in cfg.__dict__.items() if k != "output_dir"}

train_gliner2(**tk, **kw)
print("KAGGLE_TRAIN_DONE")
