---
license: apache-2.0
base_model: CohereLabs/tiny-aya-base
datasets:
  - tiny-aya-translate/tr-hi-mimi-encoded
language:
  - tr
  - hi
tags:
  - speech-to-speech-translation
  - speech-translation
  - lora
  - moshi
  - mimi
  - tinyaya
pipeline_tag: audio-to-audio
metrics:
  - bleu
---

# TinyAya Stage 2 — Turkish ↔ Hindi Speech-to-Speech Translation (LoRA)

Stage-2 simultaneous-translation adapter for Turkish↔Hindi **speech-to-speech**
translation, trained on TPU v6e-8. This repo ships the authors' **trained
deltas only** — a LoRA adapter over `CohereLabs/tiny-aya-base` plus the
custom projection / Moshi depth-decoder / audio-head / embedding tensors.

> ## ⚠️ License scope — read first
> The **`apache-2.0`** license declared above covers **only the trained
> weights in this repo** (the LoRA adapter + custom heads) and the authors'
> code. It does **NOT** relicense the components this model builds on, which
> keep their own licenses (mirrored from `THIRD_PARTY_NOTICES.md`):
> - **`CohereLabs/tiny-aya-base`** base weights → **Cohere model license**
>   (NOT Apache). Not included here; you must obtain it from Cohere and
>   comply with its terms.
> - **Moshi / Mimi** (depth decoder + audio codec) → **MIT**.
> - **FLEURS** source data → **CC BY 4.0**.

## What this is

| | |
|---|---|
| Task | TR↔HI speech-to-speech translation (Moshi inner-monologue, hierarchical codebook decode) |
| Base | `CohereLabs/tiny-aya-base` + Moshi depth decoder + Mimi codec |
| Method | LoRA (+ trained projection/heads/embeds), bf16, FSDPv2 SPMD |
| Hardware | Cloud TPU v6e-8 (single host), via Google TRC |
| Steps | 15,000 (effective batch 256) |
| Data | `tiny-aya-translate/tr-hi-mimi-encoded` (Mimi-encoded FLEURS TR/HI) |

## Evaluation

Speech-translation quality is measured with **ASR-BLEU** (Whisper transcribes
the generated target audio, BLEU vs. reference target text) and **DNSMOS**
(naturalness). Run with `scripts/eval_release.py`; numbers are reported on the
linked W&B run.

| Metric | tr→hi | hi→tr | overall |
|--------|-------|-------|---------|
| ASR-BLEU | _TBD_ | _TBD_ | _TBD_ |
| DNSMOS (ovrl) | _TBD_ | _TBD_ | _TBD_ |

## Intended use & limitations

- **Intended**: research on low-resource speech-to-speech translation and
  simultaneous translation; a Stage-2 checkpoint, not a production system.
- **Limitations**: trained on read-speech (FLEURS) — expect degradation on
  spontaneous/noisy audio; two language directions only; generation is
  autoregressive and not optimized for latency here.
- **History (transparency)**: the run was a spot v6e-8 (preemptible); an
  earlier run had a checkpoint GCS-path bug (fixed) and three W&B metrics
  (per-codebook loss, grad-norm, HBM) that logged as zero (fixed in the
  current run).

## Inference quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "CohereLabs/tiny-aya-base"  # obtain per Cohere license
ADAPTER = "tiny-aya-translate/tinyaya-stage2-tr-hi"  # this repo

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(base, ADAPTER)  # loads adapter_model.safetensors

# Then attach the custom heads (projection / depth_decoder / audio_heads /
# text_embed / model_audio_embed *.safetensors) and the Mimi codec; see
# scripts/eval_stage2.py for the full composite + generation loop.
```

The full speech→speech pipeline (Mimi encode → backbone+depth-decoder →
Mimi decode) is in `scripts/eval_stage2.py` (`ar_generate`).

## Links

- **Training run (W&B)**: _TBD (new release run)_
- **Checkpoints (GCS)**: `gs://tinyaya-stage2-tpu/checkpoints/stage2-tpu-v6e-v2/`
- **Dataset**: https://huggingface.co/datasets/tiny-aya-translate/tr-hi-mimi-encoded
- **Code**: https://github.com/tiny-aya-simulatenous-translation/tinyaya-stage2-scale

## Acknowledgments

Cloud TPU compute provided by Google's **TPU Research Cloud (TRC)**. See
`NOTICE` and `THIRD_PARTY_NOTICES.md`.
