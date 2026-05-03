#!/bin/bash
# Create the Queued Resource that provisions the TPU v4-64 slice and runs
# startup_script.sh on every host.
#
# Run from your local workstation (after setup_gcp.sh has succeeded).
#
# Configuration precedence (highest first):
#   1. shell env vars (e.g. `PROJECT_ID=foo bash launch_qr.sh`)
#   2. <repo-root>/.env  (auto-sourced)
#   3. defaults below

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# shellcheck source=./_lib.sh
source "$SCRIPT_DIR/_lib.sh"
load_env_file "$ENV_FILE"

PROJECT_ID="${PROJECT_ID:-ml-pipelines-315702}"
ZONE="${ZONE:-us-central2-b}"
QR_NAME="${QR_NAME:-tinyaya-stage2-qr}"
NODE_ID="${NODE_ID:-tinyaya-stage2}"
ACCEL_TYPE="${ACCEL_TYPE:-v4-64}"
RUNTIME="${RUNTIME:-tpu-ubuntu2204-base}"
CONFIG_FILE="${CONFIG_FILE:-configs/stage2_tpu.yaml}"
# SPOT=1  -> request preemptible (spot) capacity. Default empty = on-demand.
SPOT="${SPOT:-}"

STARTUP_SCRIPT="$SCRIPT_DIR/startup_script.sh"

if [ ! -f "$STARTUP_SCRIPT" ]; then
    echo "ERROR: $STARTUP_SCRIPT not found"
    exit 1
fi

extra_flags=()
if [ -n "$SPOT" ] && [ "$SPOT" != "0" ] && [ "$SPOT" != "false" ]; then
    extra_flags+=(--spot)
    tier="spot (preemptible)"
else
    tier="on-demand"
fi

echo "==> creating Queued Resource"
echo "    project:        $PROJECT_ID"
echo "    zone:           $ZONE"
echo "    QR name:        $QR_NAME"
echo "    node id:        $NODE_ID"
echo "    accelerator:    $ACCEL_TYPE"
echo "    runtime:        $RUNTIME"
echo "    tier:           $tier"
echo "    startup script: $STARTUP_SCRIPT"
echo "    config file:    $CONFIG_FILE"
echo

gcloud compute tpus queued-resources create "$QR_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --node-id="$NODE_ID" \
    --accelerator-type="$ACCEL_TYPE" \
    --runtime-version="$RUNTIME" \
    --metadata-from-file=startup-script="$STARTUP_SCRIPT" \
    --metadata="config-file=$CONFIG_FILE" \
    "${extra_flags[@]}"

echo
echo "==> QR submitted. Watch with:  QR_NAME=$QR_NAME bash scripts/tpu/ops.sh status"
