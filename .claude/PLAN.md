# PLAN — Public-release 15k-step TR↔HI run (TinyAya Stage 2)

> Active goal. Supersedes the completed TPU-optimization phase plan
> (full history in `.claude/PROGRESS.md` and `.claude/orchestration/`;
> its one carried-forward item — `eval_stage2.py` ASR-BLEU + DNSMOS —
> is folded into Phase C below). Edited via `/plan` / `update-plan`.

## Goal

Run a **new 15,000-step v6e-8 run** whose logs, metrics, checkpoints,
eval, and docs are complete and clean enough to be **publicly released
on both W&B and HuggingFace Hub** (model repo). Close all 15 gaps from
the 2026-06-23 release audit so the published run is credible (eval +
audio), reproducible (config + env + git + deps), and safe (no secrets,
LoRA-deltas-only, correct licensing). The prior run `b7fr72u5` is
**retired** (kept private/deprecated, not published); the new run is the
public artifact.

Builds on merged PR #5 (per-codebook loss, grad-norm, HBM telemetry —
validated live) and the just-proven safetensors/W&B-artifact path.

## Design decisions (locked with user 2026-06-23)

- **Validation = inline, TPU-safe.** Rewrite `run_validation` to
  accumulate on-device (XLA tensors, materialise once) instead of 4
  `.item()` calls + boolean-indexing; drop the `not is_tpu` gate so
  `val/loss` + `val/per_codebook_loss` log every `val_every`.
- **Audio + ASR-BLEU + DNSMOS = post-training eval job**, OFF the SPMD
  path (autoregressive generation is hostile to SPMD-TPU). **Reuse the
  existing eval scripts** — do NOT reinvent:
  - `scripts/eval_stage2.py` (ASR-BLEU + DNSMOS).
  - `scripts/eval_checkpoint.py` (Whisper ASR → sacrebleu; generates
    target-GT / teacher-forced / autoregressive wavs).
  Wire their outputs to the SAME W&B run (audio triplets, BLEU, DNSMOS,
  translation table).
- **transformers stays 4.49.x** (`[[transformers-version-ceiling]]`).

## Definition of Done

1. New run: 15000/15000 on v6e-8, exit 0, final ckpt in GCS, adapter
   saved as `.safetensors` (not `.bin`).
2. W&B run non-zero: per-codebook loss, grad-norm, peak_gb, **and**
   `val/loss` curve.
3. W&B run has full `config` + env block (torch/torch-xla/libtpu/TPU
   type/dataset rev/splits) + git SHA + tags; final ckpt registered as a
   versioned **model artifact** (safetensors).
4. Post-training eval logged to the run: **ASR-BLEU**, **DNSMOS**, ≥8
   audio triplets, translation `wandb.Table`.
5. Sanitized `train.log` in GCS (secret scan = 0).
6. `MODEL_CARD.md` + inference quickstart committed; license = LoRA
   deltas only; FLEURS CC-BY + Cohere base-model caveat stated.
7. Secret scan clean across log, `wandb.config`, checkpoint metadata.
8. **HuggingFace model repo published**: adapter safetensors + custom
   heads + `adapter_config.json` + HF README card (frontmatter:
   `base_model`, `datasets`, `license`, metrics, tags) + inference
   snippet; W&B run ↔ HF repo cross-linked.
9. `b7fr72u5` retired (private/deprecated, excluded from release).
10. `/verify` green; merged via PR before launch.

## Phase A — In-run instrumentation (train script + launch)  [code]

- [ ] **#8/#6 env+git → `wandb.config`** at `train_hierarchical.py:1076/1097`:
      torch/torch_xla/libtpu/transformers/peft/numpy versions; TPU
      topology (v6e-8, mesh); dataset `tr-hi-mimi-encoded` + revision;
      split counts (1,178,302 / 62,036); `git_sha`/`git_dirty` from a
      `GIT_SHA` env injected by the launcher (VM isn't a git checkout).
- [ ] **#13 tags/notes** on `wandb.init`.
- [ ] **#9 hygiene guard**: assert `wandb.config` = YAML only; no
      `*TOKEN*/*KEY*/*API*` keys.
- [ ] **#5/#12 log upload**: launcher sanitizes (reuse 2026-06-23
      redaction) + uploads the tee'd log → `gs://.../train_15k.log` at
      end-of-run.
- [ ] **#7 safetensors save**: `src/training/checkpointing.py` final save
      uses PEFT `safe_serialization=True`; consider converting the `.pt`
      side-tensors (audio_heads/projection/depth_decoder/embeds) too.
- [ ] **#7 artifact auto-register** at final save (reuse
      `/tmp/publish_artifact.py` logic; LoRA-deltas-only metadata+aliases).

## Phase B — TPU-safe validation rewrite  [code, moderate]

- [ ] Rewrite `run_validation` (`train_hierarchical.py:544`) with XLA
      accumulators (loss, per-cb loss, cb0-acc), no per-batch `.item()`,
      no dynamic boolean-indexing; materialise once after a FIXED N
      static-shape batches (avoid recompiles).
- [ ] Drop `and not is_tpu` at `:1816`; confirm `val/loss` +
      `val/per_codebook_loss_*` log every `val_every` with throughput
      regression within budget (fallback: end-of-run-only val).
- [ ] Persist `best_val_loss` in checkpoint `extra_state` (already wired).

## Phase C — Post-training eval, wired to W&B  [reuse existing scripts]

- [ ] Run `eval_stage2.py` / `eval_checkpoint.py` against
      `step_015000_final` off-SPMD.
- [ ] **#3 ASR-BLEU** (Whisper transcribe generated → sacrebleu) →
      `eval/asr_bleu`.
- [ ] **DNSMOS** (speechmos, in the `eval` extra) → `eval/dnsmos`.
- [ ] **#2 audio** ≥8 triplets (source / target-GT / generated) as
      `wandb.Audio`; cb0-accuracy.
- [ ] Translation `wandb.Table` (src, ref, hyp, per-row BLEU).
- [ ] Log all to the SAME run via `wandb.init(id=<run>, resume="allow")`.
- [ ] **Eval compute = A100 40/80 GB (decided 2026-06-23).** 8B backbone
      fits fp16 (~16 GB) alongside Mimi/Moshi + Whisper — no quant, no
      sequential staging, generation faithful to TPU-bf16 training.
      Subset ~100-200 BLEU samples + ~16 audio demos. Spin up on demand;
      tear down after eval.

## Phase D — Documentation  [docs]

- [ ] **#14 `MODEL_CARD.md`**: intended use, training data (FLEURS CC-BY
      + `tr-hi-mimi-encoded` rev), recipe, eval (ASR-BLEU/DNSMOS +
      curves), limitations, preemption/path-bug history, license scope.
- [ ] **#15 inference quickstart**: load `CohereLabs/tiny-aya-base` +
      apply published LoRA adapter + Mimi/Moshi; minimal decode demo.
- [ ] **#11 license**: restate LoRA-deltas-only; link `THIRD_PARTY_NOTICES.md`.

## Phase E — Release / ops  [user-gated]

- [ ] Launch new run (`stage2_tpu_v6e_v2.yaml`, `log_grad_norm: true`).
- [ ] **#10 GCS ACL**: curated public `release/` prefix (exclude
      `profiles/`, `wandb-rendezvous/`, optimizer states) vs. whole
      bucket — user decision.
- [ ] **#13 W&B**: set project/run public; author a W&B Report.
- [ ] Final secret sweep across all published surfaces.

## Phase F — HuggingFace Hub release  [new]

- [ ] Create HF model repo under **`tiny-aya-translate/`** (decided).
- [ ] Upload **LoRA adapter safetensors** + `adapter_config.json` +
      custom heads (projection / depth_decoder / audio_heads / embeds,
      as safetensors) — **no base weights** (Cohere license).
- [ ] HF `README.md` model card with YAML frontmatter: `base_model:
      CohereLabs/tiny-aya-base`, `datasets: tiny-aya-translate/tr-hi-mimi-encoded`,
      **`license: apache-2.0`** (covers the trained adapter+heads only),
      `tags`, `metrics` (ASR-BLEU, DNSMOS), `pipeline_tag`. The card MUST
      prominently state that Apache-2.0 applies only to the authors'
      trained deltas, and **mirror upstream licenses** for the rest
      (Cohere base → Cohere license; Mimi/Moshi → MIT; FLEURS → CC-BY) —
      consistent with `THIRD_PARTY_NOTICES.md`.
- [ ] Inference snippet: load base + apply adapter + Mimi/Moshi decode.
- [ ] Upload via `huggingface_hub` (HF_TOKEN from `.env`).
- [ ] Cross-link: W&B run notes → HF repo; HF card → W&B run + GCS.

## 15-item coverage matrix

Status legend: ✅ code in PR `feat/release-instrumentation-15k` (runs at
launch / post-run); ⏳ runtime/ops step still user-gated.

| # | Item | Phase | Status |
|---|------|-------|--------|
| 1 | Validation metrics | B | ✅ run_validation rewritten + ungated |
| 2 | Audio samples | C | ✅ eval_release.py (run post-train on GPU) |
| 3 | ASR-BLEU + DNSMOS | C | ✅ eval_release.py wraps eval_stage2.py |
| 4 | per-cb / grad / peak zeros | — | ✅ done (PR #5) |
| 5 | train.log → GCS | A | ✅ launch_release.sh end-of-run upload |
| 6 | git SHA linkage | A | ✅ GIT_SHA → wandb.config |
| 7 | safetensors artifact | A | ✅ final save + publish_release.py |
| 8 | env capture | A | ✅ release_meta in wandb.config |
| 9 | secret hygiene | A | ✅ config guard + log sanitize |
| 10 | GCS ACL / curated copy | E | ⏳ decision pending |
| 11 | LoRA-only license | D | ✅ MODEL_CARD scope stated |
| 12 | scrub internal ids | A | ✅ launcher log redaction |
| 13 | W&B tags/notes | A | ✅ (Report+public visibility ⏳ in E) |
| 14 | MODEL_CARD.md | D | ✅ written |
| 15 | inference quickstart | D | ✅ in MODEL_CARD |

## Risks & open decisions

- **Val rewrite recompiles** → fall back to end-of-run-only validation.
- **Eval needs CUDA** (Whisper/DNSMOS) → CPU fallback vs. GPU box (#C).
- **GCS public exposure (#10)** → default to curated `release/` prefix.
- **Audio quality at 15k** → frame card as a stage-2 research checkpoint.
- **Cost** → clean 15k ≈ 22 h v6e-8 spot; preemption plan per
  `tpu-orchestrate`.
- **Old run b7fr72u5: RETIRE (decided 2026-06-23).** Keep private/
  deprecated, exclude from the public release; do NOT delete until the
  new run + its artifact are confirmed good (it's the only trained model
  until then). The artifact `tinyaya-stage2-tr-hi-v6e-v2:v0` logged to it
  is superseded by the new run's artifact.
- **HF license (decided):** `license: apache-2.0` scoped to the trained
  adapter+heads ONLY; mirror upstream licenses for base/codec/data. Card
  must make the scope unambiguous (never imply base weights are Apache).
- **HF org (decided):** `tiny-aya-translate`.
