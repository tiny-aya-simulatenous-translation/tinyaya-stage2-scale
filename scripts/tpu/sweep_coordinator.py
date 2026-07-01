"""Multi-host sweep COORDINATOR for a single v6e-16 SPMD slice.

WHY THIS EXISTS
---------------
On the single v6e-16 the 4 hosts form ONE 16-chip SPMD mesh, so a plain per-host
``wandb agent`` desyncs: each host's agent pulls a DIFFERENT trial and the 4
hosts would run different hyperparams, corrupting the mesh. Each trial must run
as ONE 16-chip job with IDENTICAL args on all 4 hosts.

This coordinator (host-0 only) drives the sweep and broadcasts each trial's args
to every host via a shared GCS control file. The per-host runner
(``sweep_host_loop.sh``, running on ALL hosts) polls that file and, when a new
trial index appears, launches ``train_hierarchical.py --sweep <identical args>``
so the 4 hosts rendezvous into one mesh (exactly like the baseline launch does).

PROTOCOL (all objects under ``--control-prefix``, e.g.
gs://tinyaya-stage2-tpu/sweep-control/scale-grid/):
  current_trial.json         {"index": N, "args": [...], "wandb_run_id": "...",
                              "sweep_id": "...", "stop": false}
  done/trial_<N>_host_<h>     empty marker written by each host when its trial
                              process exits (coordinator waits for all hosts)
When the sweep is exhausted the coordinator writes ``{"stop": true}`` so the host
loops exit cleanly.

TWO STAGES
----------
- ``--stage grid``  : enumerate the cartesian product of the sweep YAML's
  ``parameters`` (values/value) -- deterministic, no W&B engine needed. Used for
  Stage 1 (structural grid: sweeps/sweep_stage2_scale_grid.yaml).
- ``--stage bayes`` : pull suggestions from the W&B sweep controller
  (``wandb.controller(sweep_id)``) for Stage 2 (lr x rank). Requires an existing
  ``wandb sweep`` id; VERIFY the controller path on the first trial at launch.

The metric for each trial is read back from the trial's W&B run summary
(``val/composite`` min) via the public API -- train_hierarchical logs it as it
already does; the coordinator passes ``WANDB_RUN_ID`` so the run id is known
in advance.

This module's PURE core (``enumerate_grid`` / ``build_trial_args`` /
``load_sweep_parameters``) is import-safe with no third-party deps so it can be
unit-tested on a CPU box; the orchestration imports (gsutil subprocess, wandb)
are lazy inside the functions that need them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time


# --------------------------------------------------------------------------
# Pure core (unit-tested; no wandb / gsutil / torch)
# --------------------------------------------------------------------------

def load_sweep_parameters(yaml_path: str) -> dict:
    """Return the ``parameters:`` block of a W&B sweep YAML."""
    import yaml

    with open(yaml_path) as f:
        spec = yaml.safe_load(f)
    params = spec.get("parameters")
    if not isinstance(params, dict):
        raise ValueError(f"{yaml_path} has no 'parameters:' mapping")
    return params


def enumerate_grid(parameters: dict) -> list[dict]:
    """Cartesian product of a sweep's grid ``parameters`` -> list of param dicts.

    Each parameter must be ``{values: [...]}`` (a swept axis) or ``{value: x}``
    (a fixed singleton). A ``{distribution: ...}`` continuous axis is NOT
    griddable and raises -- those belong to the bayes stage.
    """
    keys: list[str] = []
    choices: list[list] = []
    for key, spec in parameters.items():
        if not isinstance(spec, dict):
            raise ValueError(f"parameter {key!r} must be a mapping, got {spec!r}")
        if "values" in spec:
            keys.append(key)
            choices.append(list(spec["values"]))
        elif "value" in spec:
            keys.append(key)
            choices.append([spec["value"]])
        else:
            raise ValueError(
                f"parameter {key!r} has neither 'values' nor 'value' "
                f"(a 'distribution' axis is not griddable -- use --stage bayes)"
            )
    return [dict(zip(keys, combo)) for combo in itertools.product(*choices)]


def build_trial_args(params: dict) -> list[str]:
    """Map a trial's params to ``train_hierarchical.py --sweep`` CLI args.

    Param KEYS are the sweep-arg names already understood by build_parser()
    (lr_lora, lora_r, lora_alpha_mult, use_rslora, target_modules,
    lora_exclude_top, ...), so each becomes ``--<key> <value>``. Booleans are
    lowercased; lists (target_modules) are JSON-encoded so the training-side
    ``_parse_target_modules`` recovers them exactly.
    """
    args: list[str] = []
    for key, val in params.items():
        flag = f"--{key}"
        if isinstance(val, bool):
            args += [flag, "true" if val else "false"]
        elif isinstance(val, (list, tuple)):
            # compact (no spaces) so a shell word-split in the host loop keeps it
            # one token: ["q_proj","v_proj"] not ["q_proj", "v_proj"].
            args += [flag, json.dumps(list(val), separators=(",", ":"))]
        else:
            args += [flag, str(val)]
    return args


# --------------------------------------------------------------------------
# GCS control-file orchestration (lazy gsutil; no python-gcs dep)
# --------------------------------------------------------------------------

def _gsutil(*argv: str, check: bool = True, capture: bool = False):
    return subprocess.run(
        ["gsutil", *argv], check=check, capture_output=capture, text=True
    )


def _publish_trial(control_prefix: str, payload: dict) -> None:
    """Write current_trial.json atomically-ish (temp then cp)."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        local = f.name
    _gsutil("cp", local, control_prefix.rstrip("/") + "/current_trial.json")


def _wait_for_hosts(control_prefix: str, index: int, num_hosts: int,
                    poll_s: int = 20, timeout_s: int = 6 * 3600) -> bool:
    """Block until every host has written its done-marker for ``index``."""
    done_glob = control_prefix.rstrip("/") + f"/done/trial_{index}_host_*"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        out = _gsutil("ls", done_glob, check=False, capture=True)
        markers = [ln for ln in out.stdout.splitlines() if ln.strip()]
        if len(markers) >= num_hosts:
            return True
        time.sleep(poll_s)
    return False


def _read_trial_metric(entity: str, project: str, run_id: str,
                       metric: str = "val/composite") -> float | None:
    """Read the trial's best metric from its W&B run summary (public API)."""
    try:
        import wandb

        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        val = run.summary.get(metric)
        return float(val) if val is not None else None
    except Exception as e:  # noqa: BLE001 -- best-effort metric read
        print(f"[coord] WARN: could not read {metric} for {run_id}: {e}", flush=True)
        return None


def _new_run_id() -> str:
    import wandb

    return wandb.util.generate_id()


def _trial_completed(control_prefix: str, index: int) -> bool:
    """True if trial ``index`` already has a completion marker in GCS.

    Lets the coordinator survive a spot reboot mid-sweep: on restart it
    re-enumerates the (deterministic) grid and skips trials already finished,
    instead of re-running the whole sweep from trial 1.
    """
    marker = control_prefix.rstrip("/") + f"/completed/trial_{index}"
    out = _gsutil("ls", marker, check=False, capture=True)
    return out.returncode == 0 and bool(out.stdout.strip())


def _mark_completed(control_prefix: str, index: int, metric: float | None) -> None:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"index": index, "metric": metric}, f)
        local = f.name
    _gsutil("cp", local, control_prefix.rstrip("/") + f"/completed/trial_{index}")


# --------------------------------------------------------------------------
# Trial loop
# --------------------------------------------------------------------------

def _iter_grid_trials(args):
    params_spec = load_sweep_parameters(args.sweep_yaml)
    grid = enumerate_grid(params_spec)
    print(f"[coord] grid stage: {len(grid)} trials from {args.sweep_yaml}", flush=True)
    for params in grid:
        yield params, None  # no server-side run pre-created for grid


def _iter_bayes_trials(args):
    """Yield (params, controller) suggestions from the W&B sweep controller.

    NB: verify this path on the FIRST trial at launch -- the controller API is
    version-sensitive. Stops after --max-trials suggestions.
    """
    import wandb

    controller = wandb.controller(args.sweep_id)
    n = 0
    while not controller.done() and n < args.max_trials:
        controller.step()
        params = {k: v.get("value", v) for k, v in (controller.search() or {}).items()}
        if not params:
            break
        n += 1
        yield params, controller


def run(args) -> int:
    entity_project = args.wandb_project.split("/")
    entity = entity_project[0] if len(entity_project) == 2 else args.wandb_entity
    project = entity_project[-1]

    trials = _iter_grid_trials(args) if args.stage == "grid" else _iter_bayes_trials(args)

    index = 0
    for params, _ctrl in trials:
        index += 1
        cli_args = build_trial_args(params)
        run_id = f"dryrun-{index}" if args.dry_run else _new_run_id()
        payload = {
            "index": index,
            "args": cli_args,
            "wandb_run_id": run_id,
            "sweep_id": args.sweep_id or "",
            "stop": False,
        }
        print(f"[coord] trial {index}: {params}", flush=True)
        print(f"[coord]   args: {' '.join(cli_args)}", flush=True)
        if args.dry_run:
            continue
        if args.max_trials and index > args.max_trials:
            break
        # Resumability: skip trials already finished before a coordinator reboot.
        if _trial_completed(args.control_prefix, index):
            print(f"[coord] trial {index} already completed -- skipping", flush=True)
            continue

        _publish_trial(args.control_prefix, payload)
        ok = _wait_for_hosts(args.control_prefix, index, args.num_hosts)
        if not ok:
            print(f"[coord] ERROR: trial {index} timed out waiting for hosts", flush=True)
            break
        metric = _read_trial_metric(entity, project, run_id, args.metric)
        _mark_completed(args.control_prefix, index, metric)
        print(f"[coord] trial {index} done -> {args.metric}={metric}", flush=True)

    if not args.dry_run:
        _publish_trial(args.control_prefix, {"index": index + 1, "stop": True})
    print(f"[coord] sweep complete ({index} trials).", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["grid", "bayes"], required=True)
    p.add_argument("--sweep-yaml", help="sweep YAML for --stage grid (parameters enumerated)")
    p.add_argument("--sweep-id", default="", help="W&B sweep id (ENTITY/PROJECT/ID) for --stage bayes")
    p.add_argument("--control-prefix", required=True,
                   help="gs://.../sweep-control/<name> shared with the host loops")
    p.add_argument("--wandb-project", default="tinyaya-stage2-tpu",
                   help="project or ENTITY/PROJECT for reading trial metrics")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--metric", default="val/composite")
    p.add_argument("--num-hosts", type=int, default=4, help="hosts on the slice (v6e-16 = 4)")
    p.add_argument("--max-trials", type=int, default=0, help="cap trials (0 = all)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the trial plan + per-trial args; write nothing, launch nothing")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "grid" and not args.sweep_yaml:
        print("ERROR: --stage grid requires --sweep-yaml", file=sys.stderr)
        return 2
    if args.stage == "bayes" and not args.sweep_id:
        print("ERROR: --stage bayes requires --sweep-id", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
