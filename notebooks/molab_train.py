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

Fetches datasets directly from GitHub (no manual upload needed).
Deploy: paste GitHub URL into molab.marimo.io, enable GPU toggle.
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")

@app.cell
def _constants():
    REPO_RAW = "https://raw.githubusercontent.com/postphotos/ijtraining/master"
    DATASETS = {
        "v6_distill (best — full distillation, 956 docs)": "v6_distill",
        "v5_dense (balanced types, 463 docs)":             "v5_dense",
        "v4_normalized (assignment-grounded, 2805 docs)":  "v4_normalized",
    }
    return DATASETS, REPO_RAW


@app.cell
def header():
    import marimo as mo
    mo.md("""
    # SCV GLiNER2 Training Console
    **Repo:** [postphotos/ijtraining](https://github.com/postphotos/ijtraining) ·
    **Model:** `fastino/gliner2-base-v1` + LoRA ·
    **Enable GPU toggle ↑ in the molab header for CUDA**
    """)
    return (mo,)


@app.cell
def device_info(mo):
    import torch as _t
    _dev = (f"CUDA — {_t.cuda.get_device_name(0)} "
            f"({_t.cuda.get_device_properties(0).total_memory // 1024**3} GB)")  \
        if _t.cuda.is_available() else "CPU  ⚠️  enable GPU in the molab header"
    mo.stat(value=_dev, label="Device")
    return


@app.cell
def controls(mo, DATASETS):
    dataset_ui = mo.ui.dropdown(
        options=list(DATASETS.keys()),
        value=list(DATASETS.keys())[0],
        label="Dataset",
    )
    epochs_ui = mo.ui.slider(start=1, stop=10, value=4,  label="Epochs")
    batch_ui  = mo.ui.slider(start=4, stop=32, step=4, value=8, label="Batch size")
    lora_r_ui = mo.ui.slider(start=8, stop=64, step=8, value=16, label="LoRA r")
    run_btn   = mo.ui.button(label="⬇ Fetch + Train", kind="success")

    mo.vstack([
        dataset_ui,
        mo.hstack([epochs_ui, batch_ui, lora_r_ui], justify="start", gap=1),
        run_btn,
        mo.callout(mo.md(
            "Dataset is fetched directly from GitHub — no upload needed. "
            "Training runs in-kernel; molab keeps the session alive. "
            "Adapter saved to `/tmp/scv_adapter/best/`."
        ), kind="info"),
    ], gap=0.5)
    return batch_ui, dataset_ui, epochs_ui, lora_r_ui, run_btn


@app.cell
def run_training(batch_ui, dataset_ui, epochs_ui, lora_r_ui, mo, run_btn, DATASETS, REPO_RAW):
    import os, pathlib, urllib.request, torch

    if not run_btn.value:
        mo.stop(False)

    slug = DATASETS[dataset_ui.value]
    tmp  = pathlib.Path("/tmp/scv_ds"); tmp.mkdir(parents=True, exist_ok=True)
    out  = pathlib.Path("/tmp/scv_adapter"); out.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        url  = f"{REPO_RAW}/datasets/{slug}/{name}"
        dest = tmp / name
        mo.output.replace(mo.md(f"Downloading `{name}`…"))
        urllib.request.urlretrieve(url, dest)
        return str(dest)

    train_path = fetch("scv_ner_train.jsonl")
    eval_path  = fetch("scv_ner_eval.jsonl")
    mo.output.replace(mo.md("✅ Datasets ready — starting training…"))

    from gliner2.training.trainer import TrainingConfig, train_gliner2

    cfg = TrainingConfig(
        output_dir=str(out),
        experiment_name=f"scv-molab-{slug}",
        num_epochs=int(epochs_ui.value),
        batch_size=int(batch_ui.value),
        eval_batch_size=16,
        gradient_accumulation_steps=2,
        encoder_lr=1e-5, task_lr=5e-4, warmup_ratio=0.1,
        fp16=False, bf16=torch.cuda.is_bf16_supported(),
        use_lora=True,
        lora_r=int(lora_r_ui.value),
        lora_alpha=float(lora_r_ui.value * 2),
        lora_dropout=0.05,
        lora_target_modules=["encoder","span_rep","classifier","count_embed","count_pred"],
        save_adapter_only=True,
        eval_strategy="epoch",
        save_total_limit=2, save_best=True,
        metric_for_best="loss", greater_is_better=False,
        logging_steps=10, seed=42, num_workers=2,
    )
    kw = {k: v for k, v in cfg.__dict__.items() if k != "output_dir"}
    train_gliner2(model_path="fastino/gliner2-base-v1",
                  train_data=train_path, eval_data=eval_path,
                  output_dir=str(out), **kw)

    files = [(f, f.stat().st_size // 1024) for f in out.rglob("*.safetensors")]
    mo.vstack([
        mo.callout(mo.md(f"✅ **Training done** — `{slug}` · {int(epochs_ui.value)} epochs"), kind="success"),
        mo.md("**Adapter files:**"),
        *[mo.md(f"- `{f}` ({sz} KB)") for f, sz in files],
        mo.callout(mo.md(
            "Download `adapter_model.safetensors` + `adapter_config.json` from "
            "`/tmp/scv_adapter/best/` via the molab file panel, then add to "
            "`adapters/` in the repo and push."
        ), kind="info"),
    ], gap=0.5)
    return


if __name__ == "__main__":
    app.run()
