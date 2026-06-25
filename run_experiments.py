"""
run_experiments.py

Orchestrates all training runs for one algorithm on one machine.
Each of the four teammates runs this script with their assigned algorithm.

Execution flow:
    Teammate 1:  python run_experiments.py --algo PPO --mode all
    Teammate 2:  python run_experiments.py --algo SAC --mode all
    Teammate 3:  python run_experiments.py --algo TD3 --mode all
    Teammate 4:  python run_experiments.py --algo A2C --mode all

    After all teammates finish:
    python evaluate.py
"""

import os
import sys
import time
import itertools
import traceback
import argparse
from train import train

# ── Constants ──────────────────────────────────────────────────────────────────
ALL_ALGORITHMS  = ["PPO", "SAC", "TD3", "A2C"]
CONDITIONS      = ["no_curriculum", "manual", "adaptive"]
SEEDS           = [0, 1]
TOTAL_TIMESTEPS = 1_000_000

HARDCORE_CONDITIONS = ["hardcore"]
HARDCORE_SEEDS      = [0, 1]


# ── Run list builders ──────────────────────────────────────────────────────────

def get_primary_runs(algo: str) -> list:
    """Returns 6 primary runs for one algorithm (3 conditions × 2 seeds)."""
    return list(itertools.product([algo], CONDITIONS, SEEDS))


def get_hardcore_runs(algo: str) -> list:
    """Returns 2 hardcore runs for one algorithm (1 condition × 2 seeds)."""
    return list(itertools.product([algo], HARDCORE_CONDITIONS, HARDCORE_SEEDS))


# ── Skip-done logic ────────────────────────────────────────────────────────────

def get_pending_runs(
    runs:           list,
    model_base_dir: str = "models",
) -> list:
    """
    Filters out runs that already have a saved final model.
    Safe to re-run after interruptions — completed runs are skipped.
    """
    pending = []
    for algo, condition, seed in runs:
        run_name   = f"{algo}_{condition}_seed{seed}"
        model_path = os.path.join(model_base_dir, run_name, "final_model.zip")
        if os.path.exists(model_path):
            print(f"[SKIP] Already done: {run_name}")
        else:
            pending.append((algo, condition, seed))
    return pending


# ── Sequential runner ──────────────────────────────────────────────────────────

def run_sequential(
    runs:      list,
    timesteps: int = TOTAL_TIMESTEPS,
) -> None:
    """
    Executes all assigned runs sequentially on a single machine.
    Catches and logs exceptions per run so one failure does not stop the batch.
    Reports per-run and total wall time.
    """
    total      = len(runs)
    completed  = 0
    failed     = []
    wall_start = time.time()

    print(f"\nStarting {total} run(s).")
    print(f"Runtime depends on hardware and algorithm.\n")

    for i, (algo, condition, seed) in enumerate(runs):
        print(f"\n[{i+1}/{total}] {algo} | {condition} | seed {seed}")
        run_start = time.time()

        try:
            train(
                algo_name = algo,
                condition = condition,
                seed      = seed,
                timesteps = timesteps,
            )
            elapsed = time.time() - run_start
            completed += 1
            print(f"  ✓ Completed in {elapsed/60:.1f} min")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            failed.append((algo, condition, seed))

    total_elapsed = time.time() - wall_start

    print(f"\n{'='*50}")
    print(f"  Completed    : {completed}/{total}")
    print(f"  Total time   : {total_elapsed/3600:.1f} h")
    if failed:
        print(f"  Failed runs:")
        for run in failed:
            print(f"    {run[0]} | {run[1]} | seed {run[2]}")
    print(f"{'='*50}")


# ── Progress report ────────────────────────────────────────────────────────────

def print_progress(algo: str) -> None:
    """Prints which runs are done and which are pending for one algorithm."""
    all_runs = get_primary_runs(algo) + get_hardcore_runs(algo)
    total    = len(all_runs)
    done     = 0

    print(f"\n{'='*60}")
    print(f"  PROGRESS REPORT — {algo}")
    print(f"{'='*60}")
    print(f"  {'Algorithm':<8} {'Condition':<18} {'Seed':<6} {'Status'}")
    print(f"  {'-'*50}")

    for algo_name, condition, seed in all_runs:
        run_name   = f"{algo_name}_{condition}_seed{seed}"
        model_path = os.path.join("models", run_name, "final_model.zip")
        exists     = os.path.exists(model_path)
        status     = "✓ Done" if exists else "○ Pending"
        if exists:
            done += 1
        print(f"  {algo_name:<8} {condition:<18} {seed:<6} {status}")

    print(f"\n  Progress: {done}/{total} runs completed")
    print(f"{'='*60}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run BipedalWalker curriculum experiments for one algorithm"
    )
    parser.add_argument(
        "--algo",
        type     = str,
        required = True,
        choices  = ALL_ALGORITHMS,
        help     = "Algorithm to run on this machine (PPO/SAC/TD3/A2C)",
    )
    parser.add_argument(
        "--mode",
        type    = str,
        choices = ["primary", "hardcore", "all", "progress"],
        default = "primary",
        help    = "Which runs to execute (default: primary)",
    )
    parser.add_argument(
        "--timesteps",
        type    = int,
        default = TOTAL_TIMESTEPS,
        help    = f"Timesteps per run (default: {TOTAL_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--skip-done",
        action  = argparse.BooleanOptionalAction,
        default = True,
        help    = "Skip already completed runs. Use --no-skip-done to force rerun.",
    )
    args = parser.parse_args()

    if args.mode == "progress":
        print_progress(algo=args.algo)
        sys.exit(0)

    if args.mode == "primary":
        runs = get_primary_runs(algo=args.algo)
    elif args.mode == "hardcore":
        runs = get_hardcore_runs(algo=args.algo)
    else:
        runs = get_primary_runs(algo=args.algo) + get_hardcore_runs(algo=args.algo)

    if args.skip_done:
        runs = get_pending_runs(runs)

    if not runs:
        print("All runs already completed.")
        print_progress(algo=args.algo)
        sys.exit(0)

    print(f"\nRuns to execute: {len(runs)}")
    run_sequential(runs, timesteps=args.timesteps)
    print_progress(algo=args.algo)