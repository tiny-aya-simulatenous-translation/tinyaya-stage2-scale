# W&B Hyperparameter Sweeps — Stage 2

Proxy-first HP search before the expensive 22 h v6e-8 run. See
`docs/next-15k-run-plan.md` §4 for the rationale and `sweep_stage2.yaml`
for the grid.

## Prereqs (Phase 3 — implemented in this PR)
- `wandb agent` passes each swept parameter as a **CLI arg** (the `${args}` in
  `sweep_stage2.yaml` `command`). `train_hierarchical.py` parses them through
  its existing CLI-override path: `lr_lora`/`lr_depth` → `optim`, `text_weight`
  → `loss`, `warmup_steps`/`weight_decay`/`max_steps` → `train`, `val_every`/
  `val_on_tpu` → `logging`, and `lora_r`/`lora_alpha_mult` → `lora.r`/`lora.alpha`
  (`alpha = lora_alpha_mult * lora_r`). `--sweep` also logs a `sweep/text_ok`
  flag (1.0 when `train/text_loss < 11.5`, i.e. the text stream is learning).
- `configs/tpu/stage2_tpu_v6e_proxy.yaml` — a small/short proxy config
  (max_frames 200, `max_steps` 600, cheap val) so each trial is cheap.

## Runbook (steps YOU take)
1. **Create the sweep** (from your workstation; needs `wandb login`):
   ```bash
   wandb sweep sweeps/sweep_stage2.yaml
   # -> prints: wandb: Created sweep with ID: <entity>/<project>/<sweep_id>
   ```
2. **Run agents on the TPU VM** (each agent runs trials sequentially). The
   agent inherits the current env (`${env}`), so **export the full TPU env
   first** — same vars as the launcher:
   ```bash
   # on the v6e-8 (tmux), as the run user:
   cd /opt/tinyaya
   export DEVICE_BACKEND=tpu PJRT_DEVICE=TPU XLA_DISABLE_FUNCTIONALIZATION=0 \
          XLA_NO_SPECIAL_SCALARS=1 TPU_STRATEGY=fsdpv2_lora PYTHONUNBUFFERED=1 \
          LD_LIBRARY_PATH="$(dirname "$(find ~/.local/share/uv/python -name libpython3.12.so.1.0 | head -1)"):$LD_LIBRARY_PATH"
   set -a; . /opt/tinyaya/.env; set +a   # HF_TOKEN / WANDB_API_KEY
   wandb agent <entity>/<project>/<sweep_id> --count 8
   ```
   On a single v6e-8 trials are sequential (one SPMD program/host) and each
   pays the ~18 min cold compile — keep `max_steps` small and let hyperband
   prune. For the LR-range test, prefer a tiny CPU/GPU proxy to avoid the
   TPU compile tax.
3. **Pick the winner** in the W&B sweep dashboard (lowest `val/audio_loss`;
   reject any trial whose `val/text_loss` ≈ ln(V) = the data bug, and any
   whose `sweep/text_ok` stayed 0).
4. **Promote to the full run** — write the winning HPs into the prod config
   with the helper (no hand-editing; it patches only the swept keys):
   ```bash
   python scripts/promote_sweep_winner.py \
       --sweep <entity>/<project>/<sweep_id> \
       --config configs/tpu/stage2_tpu_v6e_v2.yaml
   # or pass HPs explicitly if you picked by eye:
   python scripts/promote_sweep_winner.py \
       --config configs/tpu/stage2_tpu_v6e_v2.yaml \
       --set lr_lora=2e-4 lr_depth=3e-4 lora_r=32 lora_alpha_mult=2 \
             text_weight=0.5 warmup_steps=300 weight_decay=1e-3
   ```
   Then launch the release run:
   ```bash
   bash scripts/tpu/launch_release.sh configs/tpu/stage2_tpu_v6e_v2.yaml
   ```

## Notes
- `method: bayes` + `early_terminate: hyperband` ≈ ASHA. Switch to
  `method: grid` for a deterministic coarse pass first if preferred.
- Flip `metric.name` to `val/loss` once the text stream learns.
