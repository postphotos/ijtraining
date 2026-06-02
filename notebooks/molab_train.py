# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "gliner2==1.3.1",
#   "gliner",
#   "peft",
#   "accelerate",
#   "onnxruntime",
#   "sentencepiece",
#   "polars",
# ]
# ///
"""SCV GLiNER2 Training Console — molab GPU.

Self-contained: no local ijnew deps. Deploy to molab.marimo.io by pasting the
GitHub URL. Enable GPU toggle in the molab header (NVIDIA RTX Pro 6000 Blackwell,
96 GB VRAM). Upload train.jsonl + eval.jsonl from this repo's datasets/ folder.

Dataset format (one JSON per line):
  {"input": "<article text>", "output": {"entities": {"person": ["name"], ...}}}
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def header():
    import marimo as mo
    mo.md("""
    # SCV GLiNER2 Training Console — molab
    Upload `train.jsonl` + `eval.jsonl` from `datasets/v6_distill/`, enable the
    **GPU toggle** in the header, then click **Start Training**.
    """)
    return (mo,)


@app.cell
def device_check(mo):
    import torch as _torch
    _dev = f"CUDA — {_torch.cuda.get_device_name(0)} ({_torch.cuda.get_device_properties(0).total_memory//1024**3} GB)" \
        if _torch.cuda.is_available() else "CPU (enable GPU in molab header ↑)"
    mo.stat(value=_dev, label="Device")
    return


@app.cell
def controls(mo):
    train_file = mo.ui.file(label="train.jsonl", filetypes=[".jsonl"], multiple=False)
    eval_file  = mo.ui.file(label="eval.jsonl (optional)", filetypes=[".jsonl"], multiple=False)
    epochs_ui  = mo.ui.slider(start=1, stop=10, value=4, label="Epochs")
    batch_ui   = mo.ui.slider(start=4, stop=32, step=4, value=8, label="Batch size")
    lora_r_ui  = mo.ui.slider(start=8, stop=64, step=8, value=16, label="LoRA r")
    run_btn    = mo.ui.button(label="Start Training", kind="success")

    mo.vstack([
        mo.hstack([train_file, eval_file], justify="start", gap=1),
        mo.hstack([epochs_ui, batch_ui, lora_r_ui], justify="start", gap=1),
        run_btn,
        mo.callout(mo.md(
            "Training runs **in-kernel** — the notebook will be busy until done. "
            "molab keeps the session alive. Adapter saved to `/tmp/scv_adapter/`."
        ), kind="warn"),
    ], gap=0.5)
    return batch_ui, epochs_ui, eval_file, lora_r_ui, run_btn, train_file


@app.cell
def run_training(batch_ui, epochs_ui, eval_file, lora_r_ui, mo, run_btn, train_file):
    import os, json, tempfile, pathlib

    log_lines = []

    if run_btn.value:
        if not train_file.value:
            mo.stop(True, mo.callout(mo.md("Upload `train.jsonl` first."), kind="danger"))

        import torch

        # write uploaded files to tmp
        tmp = pathlib.Path(tempfile.mkdtemp())
        train_path = tmp / "train.jsonl"
        train_path.write_bytes(train_file.value[0].contents)
        eval_path = None
        if eval_file.value:
            eval_path = tmp / "eval.jsonl"
            eval_path.write_bytes(eval_file.value[0].contents)

        out_dir = pathlib.Path("/tmp/scv_adapter")
        out_dir.mkdir(parents=True, exist_ok=True)

        from gliner2.training.trainer import TrainingConfig, train_gliner2

        cfg = TrainingConfig(
            output_dir=str(out_dir),
            experiment_name="scv-molab",
            num_epochs=int(epochs_ui.value),
            batch_size=int(batch_ui.value),
            eval_batch_size=16,
            gradient_accumulation_steps=2,
            encoder_lr=1e-5,
            task_lr=5e-4,
            warmup_ratio=0.1,
            fp16=False,
            bf16=torch.cuda.is_bf16_supported(),
            use_lora=True,
            lora_r=int(lora_r_ui.value),
            lora_alpha=float(lora_r_ui.value * 2),
            lora_dropout=0.05,
            lora_target_modules=["encoder", "span_rep", "classifier", "count_embed", "count_pred"],
            save_adapter_only=True,
            eval_strategy="epoch" if eval_path else "no",
            save_total_limit=2,
            save_best=bool(eval_path),
            metric_for_best="loss",
            greater_is_better=False,
            logging_steps=10,
            seed=42,
            num_workers=2,
        )
        tk = {"model_path": "fastino/gliner2-base-v1", "train_data": str(train_path),
              "output_dir": str(out_dir)}
        if eval_path:
            tk["eval_data"] = str(eval_path)
        kw = {k: v for k, v in cfg.__dict__.items() if k != "output_dir"}

        train_gliner2(**tk, **kw)
        log_lines.append("✅ Training complete — adapter at /tmp/scv_adapter/")

        # list adapter files
        files = list(out_dir.rglob("*.safetensors"))
        for f in files:
            log_lines.append(f"  {f} ({f.stat().st_size//1024} KB)")

    mo.vstack([mo.md(line) for line in log_lines]) if log_lines else mo.md("")
    return


if __name__ == "__main__":
    app.run()
