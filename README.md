# ijtraining

GLiNER2 NER training for the SCV History project. Self-contained — no dependency on the main `indiejones` repo.

## What's here

```
datasets/
  v4_normalized/   — assignment-grounded labels from live .com content (2,805 docs)
  v5_dense/        — dense phase-E labels, balanced types (463 docs, 3.7 ents/doc)
  v6_distill/      — full distillation, all base-model extractions (956 docs)

adapters/
  v4/best/         — LoRA adapter from v4 data (12.5 MB)
  v5/best/         — LoRA adapter from v5 data
  v6/best/         — LoRA adapter from v6 data

ground_truth/
  scv_people.txt   — 962 curated SCV person names
  scv_places.txt   — 125 curated SCV place names
  scv_things.txt   — 652 curated SCV things/topics
  label_corrections.json — 130 reviewed type-correction rules

notebooks/
  molab_train.py   — marimo training console for molab.marimo.io (free GPU)
  molab_ner.py     — marimo inference console (full-doc chunked extraction)

kernels/
  kaggle_train.py        — Kaggle GPU kernel (T4, use --accelerator NvidiaTeslaT4)
  kernel-metadata.json   — Kaggle kernel manifest
```

## Train on molab (free — NVIDIA RTX Pro 6000, 96 GB VRAM)

1. Go to [molab.marimo.io](https://molab.marimo.io)
2. New notebook → **Sync from GitHub** → paste:
   ```
   https://github.com/postphotos/ijtraining/blob/main/notebooks/molab_train.py
   ```
3. Enable **GPU** toggle in the molab header
4. Upload `datasets/v6_distill/train.jsonl` + `eval.jsonl`
5. Click **Start Training** — adapter saved to `/tmp/scv_adapter/`

## Run inference on molab (free GPU)

Same flow but use `notebooks/molab_ner.py`. Paste text or upload `docs.jsonl`
(format: `{"url":"...","text":"..."}`).

## Train on Kaggle (free T4 GPU)

```bash
# from indiejones repo
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
from src.ijnew.notebooks.ij717_logic import push_dataset, push_kernel
push_dataset('datasets/v6_distill/scv_ner_train.jsonl', 'datasets/v6_distill/scv_ner_eval.jsonl', 'musicman5821/scv-ner-v6-distill')
push_kernel('musicman5821/scv-gliner2-train', 'musicman5821/scv-ner-v6-distill', epochs=4, batch=8, accelerator='NvidiaTeslaT4')
"
```

Or push `kernels/` directly:
```bash
kaggle kernels push -p kernels/ --accelerator NvidiaTeslaT4
```

## Dataset format

```json
{"input": "Article text here...", "output": {"entities": {"person": ["Frank Sinatra"], "location": ["Newhall"]}}}
```

## Key finding: fine-tuning vs base

After 4 training runs (v4→v6), base gliner2 consistently extracts more entities
than any fine-tune on our small corpus (~1500 docs). **Use base model + editor
taxonomy reconciliation** for production extraction. Fine-tuning is reserved for
when we have 10K+ fully-labeled docs.

The real extraction lever is **full-doc chunking** (1400-char windows, no truncation)
— this takes a doc from 16 to 59 entities (the "Suddenly" film article benchmark).
