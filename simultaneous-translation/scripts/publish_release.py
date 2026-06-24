"""Publish a final checkpoint: convert all weights to safetensors, then
register a versioned W&B model artifact and (optionally) push to the
HuggingFace Hub.

Covers audit item #7 (safetensors artifact) and Phase F (HF release).
Publishes the authors' trained deltas ONLY -- the LoRA adapter + custom
heads (projection / depth_decoder / audio_heads / text & audio embeds).
The base Cohere weights are NEVER included (Cohere license).

Usage::

    # W&B artifact only
    python scripts/publish_release.py --checkpoint /tmp/ckpt_final/step_015000_final \\
        --wandb_run_id <id>

    # also push to HuggingFace
    python scripts/publish_release.py --checkpoint <dir> --wandb_run_id <id> \\
        --hf_repo tiny-aya-translate/tinyaya-stage2-tr-hi --push_hf
"""
import argparse
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# .pt state-dicts to convert to .safetensors for the published bundle.
SIDE_TENSORS = [
    "projection.pt",
    "depth_decoder.pt",
    "audio_heads.pt",
    "text_embed.pt",
    "model_audio_embed.pt",
]

ARTIFACT_NAME = "tinyaya-stage2-tr-hi-v6e-v2"


def _convert_state_dict_pt(pt_path: Path) -> Path:
    """torch.save state-dict -> safetensors (clone to drop shared storage)."""
    sd = torch.load(pt_path, map_location="cpu", weights_only=True)
    nontensor = {k: type(v).__name__ for k, v in sd.items() if not isinstance(v, torch.Tensor)}
    assert not nontensor, f"{pt_path.name}: non-tensor entries {nontensor}"
    clean = {k: v.contiguous().clone() for k, v in sd.items()}
    out = pt_path.with_suffix(".safetensors")
    save_file(clean, out)
    assert set(load_file(out)) == set(clean), f"{out.name}: round-trip mismatch"
    return out


def convert_checkpoint(ckpt: Path) -> None:
    """Convert adapter .bin + side .pt -> safetensors; drop the pickles."""
    adapter_bin = ckpt / "peft_adapter" / "adapter_model.bin"
    adapter_safe = ckpt / "peft_adapter" / "adapter_model.safetensors"
    if adapter_bin.exists() and not adapter_safe.exists():
        sd = torch.load(adapter_bin, map_location="cpu", weights_only=True)
        clean = {k: v.contiguous().clone() for k, v in sd.items()}
        save_file(clean, adapter_safe, metadata={"format": "pt"})
        adapter_bin.unlink()
        print(f"[publish] adapter: {len(clean)} tensors -> safetensors (dropped .bin)")
    for name in SIDE_TENSORS:
        p = ckpt / name
        if p.exists():
            out = _convert_state_dict_pt(p)
            p.unlink()
            print(f"[publish] {name} -> {out.name}")


def log_wandb_artifact(ckpt: Path, run_id: str, project: str, entity: str) -> None:
    import wandb

    run = wandb.init(
        project=project, entity=entity, id=run_id, resume="allow", job_type="publish-artifact"
    )
    art = wandb.Artifact(
        ARTIFACT_NAME,
        type="model",
        description=(
            "TinyAya Stage 2 TR<->HI speech-to-speech. LoRA adapter (safetensors) + "
            "projection / depth_decoder / audio_heads / embeds. NO base weights."
        ),
        metadata={
            "base_model": "CohereLabs/tiny-aya-base",
            "dataset": "tiny-aya-translate/tr-hi-mimi-encoded",
            "adapter_format": "safetensors",
            "weights_published": "LoRA deltas + custom heads only (no base weights)",
        },
    )
    art.add_dir(str(ckpt))
    run.log_artifact(art, aliases=["final", "v6e-v2"])
    art.wait()
    print(f"[publish] W&B artifact {art.name}:{art.version} -> run {run_id}")
    run.finish()


def push_hf(
    ckpt: Path, repo_id: str, card_path: Path | None, private: bool = True, tag: str | None = None
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    if card_path and card_path.exists():
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["*.bin", "*.pt"],  # safetensors only; never base weights
        commit_message=f"Upload {ckpt.name}" + (f" ({tag})" if tag else ""),
    )
    print(f"[publish] pushed to https://huggingface.co/{repo_id} (private={private})")
    if tag:
        # Git tag = the version users pin via revision=<tag>.
        api.create_tag(repo_id, tag=tag, repo_type="model", exist_ok=True)
        print(f"[publish] created tag {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="local final checkpoint dir")
    ap.add_argument("--wandb_run_id", default=None)
    ap.add_argument("--wandb_project", default="tinyaya-stage2-tpu")
    ap.add_argument("--wandb_entity", default="cataluna84")
    ap.add_argument("--hf_repo", default="tiny-aya-translate/tinyaya-stage2-tr-hi")
    ap.add_argument("--model_card", default=None, help="path to README/MODEL_CARD.md")
    ap.add_argument("--push_hf", action="store_true")
    ap.add_argument("--public", action="store_true", help="create a PUBLIC repo (default: private)")
    ap.add_argument("--tag", default=None, help="git tag for this version, e.g. v0.1.0")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    convert_checkpoint(ckpt)

    if args.wandb_run_id:
        log_wandb_artifact(ckpt, args.wandb_run_id, args.wandb_project, args.wandb_entity)
    if args.push_hf:
        card = Path(args.model_card) if args.model_card else None
        push_hf(ckpt, args.hf_repo, card, private=not args.public, tag=args.tag)


if __name__ == "__main__":
    main()
