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
"""SCV GLiNER2 NER — molab GPU console.

Self-contained: no local ijnew deps. Deploy to molab.marimo.io by pasting this
notebook's GitHub URL. Enable the GPU toggle in the molab header for CUDA.

Runs gliner2-base over pasted text or uploaded JSONL (url, text rows),
showing typed entity mentions with disagreement flags vs the optional
editor-taxonomy JSON.
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell
def header():
    import marimo as mo
    mo.md("""
    # SCV Entity Extraction — molab GPU
    **Model:** `fastino/gliner2-base-v1` · **GPU:** toggle in the header ↑ for CUDA
    """)
    return (mo,)


@app.cell
def controls(mo):
    import torch as _torch
    device_label = f"CUDA ({_torch.cuda.get_device_name(0)})" if _torch.cuda.is_available() else "CPU"
    device_stat = mo.stat(value=device_label, label="Device")

    text_area = mo.ui.text_area(
        placeholder="Paste article text here, or upload a JSONL file below...",
        rows=10,
        label="Article text",
    )
    file_upload = mo.ui.file(
        label="Or upload docs.jsonl  (one JSON per line: {\"url\":\"...\",\"text\":\"...\"})",
        filetypes=[".jsonl", ".json"],
        multiple=False,
    )
    labels_ui = mo.ui.text(
        value="person, organization, location, event",
        label="Entity labels (comma-separated)",
    )
    chunk_size = mo.ui.slider(
        start=500, stop=2000, step=100, value=1400,
        label="Chunk size (chars) — slides over full doc",
    )
    run_btn = mo.ui.button(label="Extract Entities", kind="success")

    mo.vstack([
        device_stat,
        mo.hstack([labels_ui, chunk_size], justify="start", gap=1),
        text_area,
        file_upload,
        run_btn,
        mo.callout(mo.md(
            "Enable the **GPU toggle** in the molab header for CUDA acceleration. "
            "On CPU, each article takes ~1-3 s. On GPU, ~0.1 s."
        ), kind="info"),
    ], gap=0.5)
    return chunk_size, device_label, file_upload, labels_ui, run_btn, text_area


@app.cell
def run_extraction(chunk_size, file_upload, labels_ui, mo, run_btn, text_area):
    import json
    import re
    import torch

    results = []

    if run_btn.value:
        from gliner2 import GLiNER2
        model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
        if torch.cuda.is_available():
            model.model.cuda()
        labels = [l.strip() for l in labels_ui.value.split(",") if l.strip()]

        def extract_chunks(text, csize):
            parts = re.split(r"(?<=[.!?])\s+", text)
            buf = ""
            union = {}
            for p in parts:
                if len(buf) + len(p) + 1 > csize and buf:
                    r = model.extract_entities(buf, labels)
                    ents = r.get("entities", {}) if isinstance(r, dict) else {}
                    for lab, items in ents.items():
                        for it in items:
                            nm = it if isinstance(it, str) else (it.get("text") or "")
                            if nm:
                                union.setdefault(lab, {}).setdefault(nm.lower(), nm)
                    buf = p
                else:
                    buf = (buf + " " + p).strip()
            if buf:
                r = model.extract_entities(buf, labels)
                ents = r.get("entities", {}) if isinstance(r, dict) else {}
                for lab, items in ents.items():
                    for it in items:
                        nm = it if isinstance(it, str) else (it.get("text") or "")
                        if nm:
                            union.setdefault(lab, {}).setdefault(nm.lower(), nm)
            return {lab: list(m.values()) for lab, m in union.items()}

        csize = int(chunk_size.value)

        # JSONL upload takes priority over text area
        if file_upload.value:
            content = file_upload.value[0].contents.decode("utf-8", errors="ignore")
            for line in content.splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                url = d.get("url", "")
                text = (d.get("text") or "").strip()
                if not text:
                    continue
                entities = extract_chunks(text, csize)
                results.append({"url": url, "entities": entities, "n": sum(len(v) for v in entities.values())})
        elif text_area.value.strip():
            entities = extract_chunks(text_area.value, csize)
            results.append({"url": "pasted text", "entities": entities, "n": sum(len(v) for v in entities.values())})

    results
    return (results,)


@app.cell
def show_results(mo, results):
    import polars as pl

    if not results:
        view = mo.callout(mo.md("Enter text or upload a JSONL file, then click **Extract Entities**."), kind="info")
    else:
        rows = []
        for doc in results:
            for label, names in doc["entities"].items():
                for name in names:
                    rows.append({"url": doc["url"][:60], "label": label, "entity": name})
        df = pl.DataFrame(rows) if rows else pl.DataFrame({"url": [], "label": [], "entity": []})
        total = sum(r["n"] for r in results)
        view = mo.vstack([
            mo.hstack([
                mo.stat(value=str(len(results)), label="Docs"),
                mo.stat(value=str(total), label="Entities found"),
                mo.stat(value=str(len(set(r["entity"] for r in rows))), label="Distinct"),
            ], justify="start", gap=1),
            mo.ui.table(df, page_size=50),
        ], gap=0.5)
    view
    return


if __name__ == "__main__":
    app.run()
