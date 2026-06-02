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
"""SCV GLiNER2 NER Inference Console — molab GPU.

Fetches docs.jsonl or uses pasted text. Full-doc chunked extraction (no
truncation — the fix that went from 16 → 59 entities on the Suddenly article).
Optionally loads a LoRA adapter from the repo.

Deploy: paste GitHub URL into molab.marimo.io, enable GPU toggle.
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")

@app.cell
def _constants():
    REPO_RAW = "https://raw.githubusercontent.com/postphotos/ijtraining/master"
    ADAPTERS = {
        "Base only (no fine-tune — strongest extractor)": None,
        "v6 distill adapter":    "adapters/v6/best",
        "v5 dense adapter":      "adapters/v5/best",
        "v4 normalized adapter": "adapters/v4/best",
    }
    ADAPTER_FILES = ["adapter_model.safetensors", "adapter_config.json", "README.md"]
    return ADAPTER_FILES, ADAPTERS, REPO_RAW


@app.cell
def header():
    import marimo as mo
    mo.md("""
    # SCV NER Inference — molab GPU
    **Repo:** [postphotos/ijtraining](https://github.com/postphotos/ijtraining) ·
    Full-doc chunked extraction · Enable **GPU toggle ↑** for CUDA
    """)
    return (mo,)


@app.cell
def device_info(mo):
    import torch as _t
    _dev = (f"CUDA — {_t.cuda.get_device_name(0)} "
            f"({_t.cuda.get_device_properties(0).total_memory // 1024**3} GB)") \
        if _t.cuda.is_available() else "CPU  ⚠️  enable GPU in molab header"
    mo.stat(value=_dev, label="Device")
    return


@app.cell
def controls(mo, ADAPTERS):
    adapter_ui  = mo.ui.dropdown(options=list(ADAPTERS.keys()),
                                  value=list(ADAPTERS.keys())[0], label="Adapter")
    labels_ui   = mo.ui.text(value="person, organization, location, event",
                              label="Entity labels")
    chunk_ui    = mo.ui.slider(start=500, stop=2000, step=100, value=1400,
                                label="Chunk size (chars)")
    text_area   = mo.ui.text_area(
        placeholder="Paste article text here…", rows=8, label="Article text")
    file_upload = mo.ui.file(
        label='Or upload docs.jsonl  ({"url":"...","text":"..."})',
        filetypes=[".jsonl", ".json"], multiple=False)
    run_btn     = mo.ui.button(label="Extract Entities", kind="success")

    mo.vstack([
        mo.hstack([adapter_ui, labels_ui, chunk_ui], justify="start", gap=1),
        text_area, file_upload, run_btn,
    ], gap=0.5)
    return adapter_ui, chunk_ui, file_upload, labels_ui, run_btn, text_area


@app.cell
def load_model(adapter_ui, mo, ADAPTERS, ADAPTER_FILES, REPO_RAW):
    import pathlib, urllib.request, torch
    from gliner2 import GLiNER2

    _model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
    if torch.cuda.is_available():
        _model.model.cuda()

    _adapter_key = ADAPTERS[adapter_ui.value]
    if _adapter_key:
        _tmp = pathlib.Path("/tmp/scv_adapter_infer"); _tmp.mkdir(parents=True, exist_ok=True)
        for _fname in ADAPTER_FILES:
            _url  = f"{REPO_RAW}/{_adapter_key}/{_fname}"
            _dest = _tmp / _fname
            try:
                urllib.request.urlretrieve(_url, _dest)
            except Exception:
                pass
        if (_tmp / "adapter_config.json").exists():
            from gliner2 import load_lora_adapter
            load_lora_adapter(_model, str(_tmp))

    mo.stat(value=adapter_ui.value.split("(")[0].strip(), label="Adapter loaded")
    return (_model,)


@app.cell
def run_extraction(chunk_ui, file_upload, labels_ui, mo, run_btn, text_area, _model):
    import json, re
    import polars as pl

    if not run_btn.value:
        mo.stop(False)

    _labels = [l.strip() for l in labels_ui.value.split(",") if l.strip()]
    _csize  = int(chunk_ui.value)

    def _extract_full(text):
        parts = re.split(r"(?<=[.!?])\s+", text)
        buf   = ""
        union = {}
        def _flush(b):
            if not b.strip():
                return
            r    = _model.extract_entities(b, _labels)
            ents = r.get("entities", {}) if isinstance(r, dict) else {}
            for lab, items in ents.items():
                for it in items:
                    nm = it if isinstance(it, str) else (it.get("text") or "")
                    if nm:
                        union.setdefault(lab, {}).setdefault(nm.lower(), nm)
        for p in parts:
            if len(buf) + len(p) + 1 > _csize and buf:
                _flush(buf); buf = p
            else:
                buf = (buf + " " + p).strip()
        _flush(buf)
        return {lab: list(m.values()) for lab, m in union.items()}

    _docs = []
    if file_upload.value:
        for _line in file_upload.value[0].contents.decode("utf-8", errors="ignore").splitlines():
            if not _line.strip():
                continue
            try:
                d = json.loads(_line)
                _docs.append((d.get("url", ""), (d.get("text") or "").strip()))
            except Exception:
                pass
    elif text_area.value.strip():
        _docs = [("pasted text", text_area.value.strip())]

    _rows = []
    for _url, _text in _docs:
        if not _text:
            continue
        for _label, _names in _extract_full(_text).items():
            for _name in _names:
                _rows.append({"url": _url[:80], "label": _label, "entity": _name})

    if not _rows:
        _view = mo.callout(mo.md("No entities found — check text or labels."), kind="warn")
    else:
        _df = pl.DataFrame(_rows)
        _view = mo.vstack([
            mo.hstack([
                mo.stat(value=str(len(_docs)),               label="Docs"),
                mo.stat(value=str(len(_rows)),               label="Mentions"),
                mo.stat(value=str(_df["entity"].n_unique()), label="Distinct"),
                mo.stat(value=str(_df["label"].n_unique()),  label="Types"),
            ], justify="start", gap=1),
            mo.ui.table(_df, page_size=100),
        ], gap=0.5)
    _view
    return


if __name__ == "__main__":
    app.run()
