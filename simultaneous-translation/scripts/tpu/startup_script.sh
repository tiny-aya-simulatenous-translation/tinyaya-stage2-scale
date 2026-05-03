#!/bin/bash
# TPU VM startup script.
# Runs as the default user on every host of the Queued Resource at boot.
# Idempotent: safe to re-run after a host reboot or `gcloud ... ssh ... -- 'sudo reboot'`.

set -euo pipefail
exec > >(tee -a /tmp/startup.log) 2>&1

# Metadata startup-script context can run with HOME unset (root, no login shell).
# Set it explicitly so `set -u` doesn't trip on $HOME later.
export HOME="${HOME:-/root}"
export USER="${USER:-$(id -un)}"

echo "=== [$(date -Is)] startup_script.sh begin on $(hostname) ==="

# ----- repo / branch knobs (override via VM metadata if needed) -----
REPO_URL="${REPO_URL:-https://github.com/tiny-aya-simulatenous-translation/tinyaya-stage2-scale.git}"
REPO_BRANCH="${REPO_BRANCH:-feat/tpu-support}"
REPO_DIR="${REPO_DIR:-/opt/tinyaya}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.13}"
HF_DATASET="${HF_DATASET:-tiny-aya-translate/fleurs-tr-hi-mimi-encoded}"
DATA_DIR="${DATA_DIR:-/mnt/data}"
SECRET_HF="${SECRET_HF:-hf-token}"
SECRET_WANDB="${SECRET_WANDB:-wandb-api-key}"
TMUX_SESSION="${TMUX_SESSION:-train}"
# Optional: GCS path to a tarball that overlays the cloned repo with locally
# uncommitted files (pyproject.toml, uv.lock, new configs). Useful when the
# launch infra is being iterated on and we don't want to push every change.
OVERLAY_GS_URI="${OVERLAY_GS_URI:-}"

read_meta() {
    local key=$1 default=$2
    curl -fsS -H 'Metadata-Flavor: Google' \
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$key" \
        2>/dev/null || echo "$default"
}
# Training config path (relative to simultaneous-translation/) is read from VM
# metadata so canary vs full run share the same image.
CONFIG_FILE="$(read_meta config-file configs/stage2_tpu.yaml)"
# Optional GCS overlay path (see OVERLAY_GS_URI section below).
OVERLAY_GS_URI="$(read_meta overlay-gs-uri '')"
echo "[startup] CONFIG_FILE=$CONFIG_FILE"
echo "[startup] OVERLAY_GS_URI=${OVERLAY_GS_URI:-<unset>}"

# ----- 1. system deps -----
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    tmux git curl ca-certificates build-essential

# ----- 2. uv (with pip fallback if astral.sh is unreachable) -----
if ! command -v uv >/dev/null 2>&1; then
    if ! curl -fLsS https://astral.sh/uv/install.sh -o /tmp/uv-install.sh; then
        echo "[startup] curl install of uv failed, falling back to pip"
        python3 -m pip install --user uv
    else
        sh /tmp/uv-install.sh
    fi
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version

# ----- 3. clone / update repo -----
sudo mkdir -p "$(dirname "$REPO_DIR")"
sudo chown "$USER:$USER" "$(dirname "$REPO_DIR")"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch --all --prune
git checkout "$REPO_BRANCH"
git pull --ff-only

cd "$REPO_DIR/simultaneous-translation"

# ----- 3.5. optional overlay of locally-modified files from GCS -----
# The infra files (pyproject.toml, uv.lock, configs/stage2_tpu*.yaml) may not
# be in the cloned commit yet. If OVERLAY_GS_URI metadata is set, fetch and
# unpack on top of the cloned repo before `uv sync`.
if [ -n "$OVERLAY_GS_URI" ]; then
    echo "[startup] fetching code overlay from $OVERLAY_GS_URI"
    gcloud storage cp "$OVERLAY_GS_URI" /tmp/overlay.tar.gz
    tar -xzvf /tmp/overlay.tar.gz -C "$REPO_DIR/simultaneous-translation"
    rm -f /tmp/overlay.tar.gz
fi

# ----- 4. python + deps via uv (frozen lockfile) -----
uv python install "$PYTHON_VERSION"
uv sync --frozen --extra eval

# ----- 5. secrets from Secret Manager -----
HF_TOKEN="$(gcloud secrets versions access latest --secret="$SECRET_HF")"
export HF_TOKEN
if WANDB_API_KEY="$(gcloud secrets versions access latest --secret="$SECRET_WANDB" 2>/dev/null)"; then
    export WANDB_API_KEY
fi

# ----- 6. dataset to /mnt/data (resumable) -----
sudo mkdir -p "$DATA_DIR"
sudo chown "$USER:$USER" "$DATA_DIR"
HF_HUB_ENABLE_HF_TRANSFER=1 \
uv run huggingface-cli download "$HF_DATASET" \
    --repo-type dataset \
    --local-dir "$DATA_DIR" \
    --resume-download

# ----- 7. launch training with auto-restart in tmux -----
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION" "
    set -euo pipefail
    cd '$REPO_DIR/simultaneous-translation'
    while true; do
        echo \"[\$(date -Is)] launching train_hierarchical.py\" | tee -a /tmp/train.log
        DEVICE_BACKEND=tpu PJRT_DEVICE=TPU \
        HF_TOKEN='$HF_TOKEN' \
        WANDB_API_KEY='${WANDB_API_KEY:-}' \
        uv run python scripts/train_hierarchical.py \
            --config '$CONFIG_FILE' \
            --resume auto 2>&1 | tee -a /tmp/train.log
        echo \"[\$(date -Is)] training exited with status \$?, sleeping 30s\" | tee -a /tmp/train.log
        sleep 30
    done
"

echo "=== [$(date -Is)] startup_script.sh complete on $(hostname) ==="
echo "Tail logs with: tmux attach -t $TMUX_SESSION   OR   tail -f /tmp/train.log"
