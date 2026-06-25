# BipedalWalker Curriculum Learning

A study of whether curriculum learning — training on progressively harder
terrain rather than full difficulty from the start — improves reinforcement
learning performance on `BipedalWalker-v3`, and whether the answer depends on
which RL algorithm is used.

For installation and environment setup, see **[SETUP.md](SETUP.md)**.

---

## Research Question

Four RL algorithms with very different internal mechanics (on-policy vs.
off-policy, replay buffer vs. none) are each trained under three different
terrain regimes. The goal is to see whether curriculum learning is
universally helpful, helpful only for certain algorithm families, or not
helpful at all once evaluated fairly.

## Experimental Design

**Algorithms:** PPO, SAC, TD3, A2C (all from Stable-Baselines3, using
rl-baselines3-zoo's tuned BipedalWalker-v3 hyperparameters)

**Primary curriculum conditions** (3):

| Condition | Description |
|---|---|
| `no_curriculum` | Standard full-difficulty terrain throughout. Baseline. |
| `manual` | Fixed two-phase schedule: terrain roughness = 0.2 for the first 300K steps, then standard terrain for the remainder. |
| `adaptive` | Terrain roughness starts at 0.2 and increases by 0.1 automatically once the rolling 10-episode mean reward crosses 100, up to a cooldown of 50 episodes between increases. |

**Secondary condition** (evaluated separately, not part of the 3-way comparison):

| Condition | Description |
|---|---|
| `hardcore` | BipedalWalker's official harder variant (stumps, pits). Used both for direct training and for zero-shot generalization testing of the best standard-terrain models. |

**Seeds:** 2 per algorithm/condition combination (seeds 0 and 1).

> **Statistical note:** 2 seeds gives an indicative measure of variance, not
> a rigorous confidence interval. This was a deliberate compute-budget
> tradeoff given 4 algorithms × 3 primary conditions × 2 seeds = 24 primary
> runs, plus 4 algorithms × 2 seeds = 8 hardcore runs (32 runs total).

**Evaluation:** Regardless of training terrain, every model is evaluated on
**standard** `BipedalWalker-v3` for a fair cross-condition comparison.
Reported scores use each run's **best checkpoint** (saved by `EvalCallback`),
not the final model — this matters because training can stop early via
`StopTrainingOnRewardThreshold` once a run solves the task (reward ≥ 300),
so final-step reward isn't comparable across runs that stopped at different
points.

---

## Key Methodological Notes

- **A2C uses observation/reward normalization (VecNormalize); PPO/SAC/TD3 do
  not.** This follows rl-baselines3-zoo's per-algorithm reference
  configuration (`hyperparams/a2c.yml` sets `normalize: true` for
  BipedalWalker-v3) rather than being an inconsistency. A2C's on-policy
  updates with no replay buffer make it considerably more sensitive to
  observation scale than the off-policy methods. **Cross-algorithm
  comparisons involving A2C should be read with this in mind**;
  within-algorithm comparisons (A2C across its own 3 conditions) are
  unaffected, since A2C receives VecNormalize in all three.

- **Phase 2 terrain uses `seed + 1`, not the same seed as Phase 1.** This is
  deliberate: reusing the same seed across the curriculum transfer could let
  a manual-curriculum model appear to generalize by memorizing the exact
  terrain layout it saw in Phase 1, rather than by learning a genuinely
  transferable gait. The offset keeps Phase 2 terrain novel while remaining
  fully deterministic across restarts.

- **Early stopping affects total training budget.** A run that solves the
  task early receives fewer total environment steps than one that doesn't.
  Reporting best-checkpoint reward makes scores comparable, but readers
  should note that total steps trained is not constant across all runs —
  see the results report for steps-to-solve alongside final reward.

---

## Repository Structure

```
.
├── README.md                  ← this file
├── SETUP.md                   ← installation & environment instructions
├── requirements.txt            ← core dependencies (torch installed separately — see SETUP.md)
├── install.ps1                 ← automated setup script (Windows/PowerShell)
│
├── simplified_terrain.py       ← SimplifiedTerrainWrapper, AdaptiveTerrainWrapper,
│                                  environment factory functions
├── train.py                    ← train() — single-run training with checkpoint/resume logic
├── run_experiments.py          ← batch runner — executes all runs for one algorithm
├── evaluate.py                 ← evaluation, results aggregation, plotting
│
├── models/                      (generated) saved checkpoints and final models per run
├── logs/                        (generated) TensorBoard training logs
├── eval_logs/                   (generated) EvalCallback evaluation logs (learning curves)
└── results/                     (generated) CSVs, comparison plots, learning curves
```

## Running the Project

Each teammate is assigned one algorithm and runs all of that algorithm's
experiments on their own machine:

```bash
python run_experiments.py --algo PPO --mode all
python run_experiments.py --algo SAC --mode all
python run_experiments.py --algo TD3 --mode all
python run_experiments.py --algo A2C --mode all
```

`--mode all` runs both the 6 primary runs (3 conditions × 2 seeds) and the
2 hardcore runs for that algorithm. Use `--mode primary` or `--mode hardcore`
to run a subset, or `--mode progress` to check completion status without
training anything. Runs are checkpointed and safely resumable — an
interrupted run picks back up from its last checkpoint rather than
restarting from scratch.

Once all teammates have finished and all `models/` directories have been
collected onto one machine:

```bash
python evaluate.py
```

This evaluates every run, generates the results table and comparison plots,
runs the hardcore zero-shot and directly-trained evaluations, and writes
everything to `results/`.

See **[SETUP.md](SETUP.md)** for environment setup before running any of
the above.

---

## Contributors
| Algorithm | GitHub |
|---|---|
| PPO | [Rodina Mohamed](https://github.com/Rodina222) |
| SAC | [Farida Gaber](https://github.com/Farida-gaber44) |
| TD3 | [Ahmed Al-Shobaki](https://github.com/ahmedsh711) |
| A2C | [Raneem Hussien](https://github.com/Raneemhussien) |

> Each contributor trained and evaluated one algorithm across all curriculum
> conditions on their own machine. Results were collected onto a single
> machine for final evaluation via `python evaluate.py`.

---
