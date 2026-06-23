---
license: apache-2.0
base_model: CohereLabs/tiny-aya-base
base_model_relation: adapter
library_name: peft
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
# Eval results go here once the GPU eval (ASR-BLEU + DNSMOS) has run; the
# Hub renders model-index as a metrics widget. Template:
# model-index:
#   - name: tinyaya-stage2-tr-hi
#     results:
#       - task: { type: audio-to-audio, name: Speech-to-Speech Translation }
#         dataset: { type: google/fleurs, name: FLEURS (tr/hi) }
#         metrics:
#           - { type: bleu, name: ASR-BLEU (tr->hi), value: TBD }
#           - { type: bleu, name: ASR-BLEU (hi->tr), value: TBD }
# When a better checkpoint supersedes this repo, point users to it with:
# new_version: tiny-aya-translate/<next-repo>
---

> **Version:** `v0.1.0` — step-15000 checkpoint (first public release; eval pending).
> Versions are git tags in this repo; load a specific one with
> `revision="v0.1.0"`. See **Version history** at the bottom.

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

## Training procedure

- **Init**: LoRA (r=16) on the `tiny-aya-base` backbone; Moshi depth decoder
  initialized from `kyutai/moshiko`; projection / per-codebook audio heads /
  audio & text embeddings trained from scratch.
- **Recipe**: effective batch 256 (b=8 × grad-accum 4 × 8 chips), bf16,
  FSDPv2 SPMD, cosine LR (lr_lora 1.5e-4), 500 warmup steps, `max_frames=300`,
  8 codebooks, hierarchical codebook loss (text_weight 0.1, audio_weight 1.0).
- **Compute**: single-host Cloud TPU v6e-8 (spot), ~22 h, 15,000 steps
  (≈ 3.3 epochs over 1,178,302 train pairs).

**Convergence note (honest):** the **audio** loss plateaued by ~step 8,000
(≈1.7 epochs) and did not improve over the final 7k steps. The **text /
inner-monologue** stream did not learn in this run (loss ≈ random), pending
a data-pipeline fix. So this checkpoint reflects the audio-translation
capability of the recipe at convergence, not an under-trained model.

## Evaluation

Speech-translation quality is measured with **ASR-BLEU** (Whisper transcribes
the generated target audio, BLEU vs. reference target text) and **DNSMOS**
(naturalness). Reproduce with `scripts/eval_release.py`. **Pending for this
release** (`v0.1.0`); will be filled in the YAML `model-index` + below.

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

## Version history

Versions are **git tags** in this repo (HF convention: one checkpoint per
repo, versioned with tags). Load a specific one with `revision=`:

```python
PeftModel.from_pretrained(base, "tiny-aya-translate/tinyaya-stage2-tr-hi", revision="v0.1.0")
```

| Version | Step | Notes |
|---------|------|-------|
| `v0.1.0` | 15,000 | First release. Audio converged ~step 8k; text stream not yet learning; eval pending. |

When a materially different model is trained (new data / architecture), it
goes in a **new repo**, and this card is updated with a `new_version:` field
so the Hub shows a banner linking forward.

## Acknowledgments

Cloud TPU compute provided by Google's **TPU Research Cloud (TRC)**. See
`NOTICE` and `THIRD_PARTY_NOTICES.md`.
