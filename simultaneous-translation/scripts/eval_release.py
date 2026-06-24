"""Phase C: run the Stage-2 ASR-BLEU + DNSMOS eval against a final
checkpoint and log the results to the training's W&B run.

This is a thin wrapper around ``eval_stage2.py`` (which already does
autoregressive generation + Whisper ASR-BLEU + DNSMOS + demo wavs). It
runs that eval, then logs BLEU, DNSMOS, audio triplets, and a
translation table to the SAME W&B run via ``resume``.

Runs OFF the SPMD-TPU path -- on a GPU box (A100 40/80 GB fits the ~8B
backbone in fp16). Train on TPU, evaluate on GPU.

Usage::

    python scripts/eval_release.py \\
        --checkpoint /tmp/ckpt_final/step_015000_final \\
        --val_split /mnt/data/splits/val.jsonl \\
        --wandb_run_id <run-id> --max_samples 200 --num_demos 8
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="local final checkpoint dir")
    ap.add_argument("--val_split", required=True)
    ap.add_argument("--encoded_dir", default=None)
    ap.add_argument("--out_dir", default="eval_outputs/release")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--num_demos", type=int, default=8)
    ap.add_argument("--num_codebooks", type=int, default=8)
    ap.add_argument("--wandb_run_id", required=True)
    ap.add_argument("--wandb_project", default="tinyaya-stage2-tpu")
    ap.add_argument("--wandb_entity", default="cataluna84")
    ap.add_argument(
        "--skip_eval", action="store_true", help="reuse an existing out_dir (don't regenerate)"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    here = Path(__file__).parent

    # 1) Canonical eval (generation + ASR-BLEU + DNSMOS + demo wavs).
    if not args.skip_eval:
        cmd = [
            sys.executable, str(here / "eval_stage2.py"),
            "--checkpoint", args.checkpoint,
            "--val_split", args.val_split,
            "--out_dir", str(out_dir),
            "--max_samples", str(args.max_samples),
            "--num_demos", str(args.num_demos),
            "--num_codebooks", str(args.num_codebooks),
        ]
        if args.encoded_dir:
            cmd += ["--encoded_dir", args.encoded_dir]
        subprocess.run(cmd, check=True)

    summary = json.loads((out_dir / "summary.json").read_text())
    per_sample = [
        json.loads(line)
        for line in (out_dir / "per_sample.jsonl").read_text().splitlines()
        if line.strip()
    ]

    # 2) Log to the SAME W&B run.
    import soundfile as sf
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.wandb_run_id,
        resume="allow",
        job_type="release-eval",
    )

    log = {"eval/bleu_overall": summary.get("bleu_overall"), "eval/n_samples": summary.get("n")}
    for direction, score in (summary.get("bleu_by_direction") or {}).items():
        log[f"eval/bleu_{direction.replace('->', '_')}"] = score
    for key in ("dnsmos_generated", "dnsmos_groundtruth"):
        st = summary.get(key)
        if st:
            for comp in ("sig", "bak", "ovrl"):
                log[f"eval/{key}_{comp}"] = st[comp][0]  # mean
    run.log(log)

    # Translation table.
    tbl = wandb.Table(columns=["direction", "reference", "hypothesis"])
    for r in per_sample[:500]:
        tbl.add_data(r.get("direction"), r.get("ref"), r.get("hyp"))
    run.log({"eval/translations": tbl})

    # Audio triplets (source / generated / ground-truth).
    n_logged = 0
    audio_log = {}
    demos_dir = out_dir / "demos"
    for direction_dir in sorted(p for p in demos_dir.glob("*") if p.is_dir()):
        for sample_dir in sorted(p for p in direction_dir.glob("*") if p.is_dir()):
            try:
                src, sr = sf.read(sample_dir / "1_source.wav")
                gen, _ = sf.read(sample_dir / "2_generated.wav")
                gt, _ = sf.read(sample_dir / "3_groundtruth.wav")
            except Exception:
                continue
            tag = f"audio/{direction_dir.name}/{sample_dir.name}"
            audio_log[f"{tag}/source"] = wandb.Audio(src, sample_rate=sr)
            audio_log[f"{tag}/generated"] = wandb.Audio(gen, sample_rate=sr)
            audio_log[f"{tag}/groundtruth"] = wandb.Audio(gt, sample_rate=sr)
            n_logged += 1
    if audio_log:
        run.log(audio_log)

    print(
        f"[eval_release] logged BLEU={summary.get('bleu_overall')}, "
        f"{n_logged} audio triplets, {len(per_sample)} rows -> run {args.wandb_run_id}"
    )
    run.finish()


if __name__ == "__main__":
    main()
