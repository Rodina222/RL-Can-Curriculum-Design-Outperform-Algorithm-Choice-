"""
evaluate.py

Model evaluation, results aggregation, and visualisation.

Design:
  evaluate_model() is the single evaluation function. It accepts run_dir
  and use_best=True (default), resolving the model path internally.
  This eliminates the previous two-function design where evaluate_model()
  loaded whatever path was given and evaluate_best_checkpoint() was a
  separate wrapper — a design that made it easy to accidentally load the
  wrong model.

  Model path resolution inside evaluate_model():
    use_best=True  → best/best_model.zip   (peak training reward) [DEFAULT]
    use_best=False → final_model.zip        (last training step, may be collapsed)
    Fallback: if best_model.zip not found, final_model.zip is used with warning.

  A2C: vecnormalize.pkl is always at run_dir level. Loaded automatically.
  Missing pkl raises FileNotFoundError — no silent degradation.

  Report note: use_best=True is always recommended. The final model may have
  collapsed after its peak due to policy degradation, entropy collapse, or
  off-policy overfitting. The best checkpoint is what the algorithm actually
  achieved and is the correct value to report for fair cross-run comparisons.

Execution flow:
    python evaluate.py      (after all teammates finish training)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── Constants ──────────────────────────────────────────────────────────────────
ALGORITHMS = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "A2C": A2C,
}

CONDITIONS      = ["no_curriculum", "manual", "adaptive"]
SEEDS           = [0, 1]
N_EVAL_EPISODES = 10

CONDITION_LABELS = {
    "no_curriculum": "No Curriculum",
    "manual":        "Manual Curriculum",
    "adaptive":      "Adaptive Curriculum",
}

CONDITION_COLORS = {
    "no_curriculum": "#2196F3",
    "manual":        "#4CAF50",
    "adaptive":      "#FF9800",
}

ALGORITHM_COLORS = {
    "PPO": "#E91E63",
    "SAC": "#9C27B0",
    "TD3": "#00BCD4",
    "A2C": "#FF5722",
}


# ══════════════════════════════════════════════════════════════════════════════
# Single model evaluation — the ONE evaluation function
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    algo_name:  str,
    run_dir:    str,
    env_id:     str  = "BipedalWalker-v3",
    hardcore:   bool = False,
    n_episodes: int  = N_EVAL_EPISODES,
    render:     bool = False,
    use_best:   bool = True,
) -> tuple[float, float]:
    """
    Evaluates a trained model and returns (mean_reward, std_reward).

    Args:
        algo_name:  One of PPO, SAC, TD3, A2C
        run_dir:    Run directory e.g. models/PPO_manual_seed0/
                    Model path is resolved internally — callers never
                    need to construct model paths manually.
        env_id:     Gymnasium environment ID
        hardcore:   If True, evaluate on BipedalWalker Hardcore
        n_episodes: Number of evaluation episodes
        render:     Render to window during evaluation
        use_best:   True  → load best/best_model.zip (default, recommended)
                    False → load final_model.zip

    Model path resolution:
        use_best=True  → {run_dir}/best/best_model.zip (peak training reward)
                         Falls back to final_model.zip with a warning if
                         best checkpoint is not found.
        use_best=False → {run_dir}/final_model.zip (last training step)

    A2C:
        Automatically loads {run_dir}/vecnormalize.pkl and wraps the eval
        env with VecNormalize(training=False). Missing pkl raises
        FileNotFoundError — no silent degradation.

    Seeding intentionally omitted: deterministic=True makes policy
    behaviour deterministic; terrain randomness averages out over 10 eps.
    """

    # ── Resolve model path internally — callers never touch paths ─────────────
    best_path  = os.path.join(run_dir, "best", "best_model")
    final_path = os.path.join(run_dir, "final_model")

    if use_best and os.path.exists(best_path + ".zip"):
        model_path = best_path
        print(f"  [✓ Best ] {best_path}.zip")
    elif os.path.exists(final_path + ".zip"):
        if use_best:
            # Best checkpoint was not saved — training may have been
            # interrupted before EvalCallback found a new best, or
            # the run ended before the first eval checkpoint.
            print(
                f"  [! Warn ] best_model.zip not found in {run_dir}/best/\n"
                f"            Falling back to final_model.zip.\n"
                f"            This may underestimate peak performance."
            )
        model_path = final_path
        print(f"  [  Final] {final_path}.zip")
    else:
        raise FileNotFoundError(
            f"No model found in {run_dir}.\n"
            f"Looked for:\n"
            f"  {best_path}.zip\n"
            f"  {final_path}.zip\n"
            f"Ensure training completed successfully."
        )

    # ── Build environment ──────────────────────────────────────────────────────
    render_mode = "human" if render else None
    tag         = " [HC]" if hardcore else ""

    if hardcore:
        raw_env = gym.make(
            "BipedalWalker-v3", hardcore=True, render_mode=render_mode
        )
    else:
        raw_env = gym.make(env_id, render_mode=render_mode)

    algo_class = ALGORITHMS[algo_name]
    model      = algo_class.load(model_path)

    # ── A2C: VecNormalize required ─────────────────────────────────────────────
    # vecnormalize.pkl is ALWAYS at run_dir level, regardless of whether
    # we loaded the best or final model. This is correct: the pkl contains
    # the running obs/reward statistics from the end of training, which is
    # the correct normalisation to apply at evaluation time regardless of
    # which model checkpoint we are evaluating.
    if algo_name == "A2C":
        vn_path = os.path.join(run_dir, "vecnormalize.pkl")

        if not os.path.exists(vn_path):
            raise FileNotFoundError(
                f"vecnormalize.pkl not found at {vn_path}.\n"
                f"A2C evaluation requires the normalisation stats saved during "
                f"training. Without them, raw unnormalised observations are fed "
                f"to a policy trained on normalised observations — results would "
                f"be meaningless. Re-run training to regenerate the file."
            )

        def _make_a2c_eval_thunk(env):
            def _thunk():
                return Monitor(env)
            return _thunk

        vec_env             = DummyVecEnv([_make_a2c_eval_thunk(raw_env)])
        vec_env             = VecNormalize.load(vn_path, vec_env)
        vec_env.training    = False    # do NOT update running stats during eval
        vec_env.norm_reward = False    # report raw rewards, not normalised

        episode_rewards = []
        for episode in range(n_episodes):
            obs   = vec_env.reset()
            done  = False
            total = 0.0
            while not done:
                action, _                = model.predict(obs, deterministic=True)
                obs, reward, done_arr, _ = vec_env.step(action)
                done   = bool(done_arr[0])
                total += float(reward[0])   # raw reward (norm_reward=False)
            episode_rewards.append(total)
            print(f"  Episode {episode+1:02d}/{n_episodes}{tag}: {total:.2f}")

        vec_env.close()
        return float(np.mean(episode_rewards)), float(np.std(episode_rewards))

    # ── All other algorithms ───────────────────────────────────────────────────
    episode_rewards = []
    for episode in range(n_episodes):
        obs, _ = raw_env.reset()
        done   = False
        total  = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = raw_env.step(action)
            done  = terminated or truncated
            total += reward
        episode_rewards.append(total)
        print(f"  Episode {episode+1:02d}/{n_episodes}{tag}: {total:.2f}")

    raw_env.close()
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


# ══════════════════════════════════════════════════════════════════════════════
# Full benchmark evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all(
    model_base_dir: str  = "models",
    env_id:         str  = "BipedalWalker-v3",
    n_episodes:     int  = N_EVAL_EPISODES,
    use_best:       bool = True,
) -> pd.DataFrame:
    """
    Evaluates all trained primary models and returns a results DataFrame.

    use_best=True (default): loads best checkpoint — recommended for report.
    use_best=False: loads final model — last training step (may be collapsed).
    """
    results = []

    for algo in ALGORITHMS.keys():
        for condition in CONDITIONS:
            seed_rewards = []

            for seed in SEEDS:
                run_name = f"{algo}_{condition}_seed{seed}"
                run_dir  = os.path.join(model_base_dir, run_name)

                # Use final_model.zip as the run-completion sentinel.
                # evaluate_model() will still load the best checkpoint
                # (when use_best=True) even though we check for final here.
                if not os.path.exists(os.path.join(run_dir, "final_model.zip")):
                    print(f"[SKIP] Run not complete: {run_name}")
                    continue

                print(f"\nEvaluating: {algo} | {condition} | seed {seed}")

                try:
                    mean, std = evaluate_model(
                        algo, run_dir,
                        env_id     = env_id,
                        n_episodes = n_episodes,
                        use_best   = use_best,
                    )
                except FileNotFoundError as e:
                    print(f"  [ERROR] {e}")
                    continue

                seed_rewards.append(mean)
                results.append({
                    "algorithm":   algo,
                    "condition":   condition,
                    "seed":        seed,
                    "mean_reward": mean,
                    "std_reward":  std,
                })

            if seed_rewards:
                print(
                    f"\n  [{algo} | {condition}] Across seeds: "
                    f"{np.mean(seed_rewards):.2f} ± {np.std(seed_rewards):.2f}"
                )

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# Hardcore trained evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_hardcore_trained(
    model_base_dir: str  = "models",
    n_episodes:     int  = N_EVAL_EPISODES,
    use_best:       bool = True,
) -> pd.DataFrame:
    """
    Evaluates models trained directly on hardcore (condition='hardcore').
    """
    print("\n" + "="*60)
    print("  HARDCORE TRAINED MODELS EVALUATION")
    print("="*60)

    results = []

    for algo in ALGORITHMS.keys():
        for seed in SEEDS:
            run_name = f"{algo}_hardcore_seed{seed}"
            run_dir  = os.path.join(model_base_dir, run_name)

            if not os.path.exists(os.path.join(run_dir, "final_model.zip")):
                print(f"[SKIP] Run not complete: {run_name}")
                continue

            print(f"\nEvaluating hardcore-trained: {algo} | seed {seed}")

            try:
                mean, std = evaluate_model(
                    algo, run_dir,
                    hardcore   = True,
                    n_episodes = n_episodes,
                    use_best   = use_best,
                )
            except FileNotFoundError as e:
                print(f"  [ERROR] {e}")
                continue

            results.append({
                "algorithm":   algo,
                "seed":        seed,
                "mean_reward": mean,
                "std_reward":  std,
            })
            print(f"  Result: {mean:.2f} ± {std:.2f}")

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# Zero-shot hardcore evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_zero_shot_hardcore(
    model_base_dir: str          = "models",
    results_df:     pd.DataFrame = None,
    n_episodes:     int          = N_EVAL_EPISODES,
    use_best:       bool         = True,
) -> pd.DataFrame:
    """
    Evaluates the best v3-trained model per algorithm zero-shot on hardcore.

    If results_df provided (from evaluate_all()), identifies the best
    run/seed from pre-computed scores without re-evaluating every model.

    use_best=True: tests best checkpoint — what the algorithm actually
    achieved, not the potentially-collapsed final policy. Recommended.
    """
    print("\n" + "="*60)
    print("  ZERO-SHOT HARDCORE EVALUATION")
    print("="*60)

    results = []

    for algo in ALGORITHMS.keys():

        # ── Identify best v3 run for this algorithm ────────────────────────────
        if results_df is not None and not results_df.empty:
            # Use pre-computed results — no re-evaluation needed
            algo_df = results_df[results_df["algorithm"] == algo]
            if algo_df.empty:
                print(f"[SKIP] No v3 results for {algo}")
                continue

            best_row       = algo_df.loc[algo_df["mean_reward"].idxmax()]
            best_condition = str(best_row["condition"])
            best_seed      = int(best_row["seed"])
            best_v3_mean   = float(best_row["mean_reward"])
            run_dir        = os.path.join(
                model_base_dir,
                f"{algo}_{best_condition}_seed{best_seed}",
            )

        else:
            # Fallback: scan and evaluate all models to find the best
            best_v3_mean   = -np.inf
            run_dir        = None
            best_condition = None
            best_seed      = None

            for condition in CONDITIONS:
                for seed in SEEDS:
                    rd = os.path.join(
                        model_base_dir,
                        f"{algo}_{condition}_seed{seed}",
                    )
                    if not os.path.exists(os.path.join(rd, "final_model.zip")):
                        continue
                    try:
                        mean, _ = evaluate_model(
                            algo, rd,
                            env_id     = "BipedalWalker-v3",
                            n_episodes = n_episodes,
                            use_best   = use_best,
                        )
                    except FileNotFoundError:
                        continue
                    if mean > best_v3_mean:
                        best_v3_mean   = mean
                        run_dir        = rd
                        best_condition = condition
                        best_seed      = seed

            if run_dir is None:
                print(f"[SKIP] No v3 models found for {algo}")
                continue

        # Verify run directory exists
        if not os.path.exists(os.path.join(run_dir, "final_model.zip")):
            print(f"[SKIP] Model files missing for {algo}")
            continue

        print(
            f"\n{algo} best v3 model: {best_condition} seed {best_seed} "
            f"(v3 mean: {best_v3_mean:.2f})"
        )
        print("Testing zero-shot on Hardcore...")

        try:
            mean_hc, std_hc = evaluate_model(
                algo, run_dir,
                hardcore   = True,
                n_episodes = n_episodes,
                use_best   = use_best,
            )
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            continue

        print(f"  Zero-shot hardcore: {mean_hc:.2f} ± {std_hc:.2f}")

        results.append({
            "algorithm":            algo,
            "best_v3_condition":    best_condition,
            "best_v3_seed":         best_seed,
            "v3_mean_reward":       best_v3_mean,
            "hardcore_mean_reward": mean_hc,
            "hardcore_std_reward":  std_hc,
        })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# Learning curve loading
# ══════════════════════════════════════════════════════════════════════════════

def load_learning_curves(eval_log_base_dir: str = "eval_logs") -> dict:
    """
    Loads EvalCallback evaluations.npz files.
    Returns dict keyed by (algo, condition, seed) →
    (timesteps, mean_rewards, std_rewards).
    """
    curves = {}

    for algo in ALGORITHMS.keys():
        for condition in CONDITIONS + ["hardcore"]:
            for seed in SEEDS:
                run_name = f"{algo}_{condition}_seed{seed}"
                npz_path = os.path.join(
                    eval_log_base_dir, run_name, "evaluations.npz"
                )
                if not os.path.exists(npz_path):
                    continue

                data         = np.load(npz_path)
                timesteps    = data["timesteps"]
                results_arr  = data["results"]
                mean_rewards = results_arr.mean(axis=1)
                std_rewards  = results_arr.std(axis=1)

                curves[(algo, condition, seed)] = (
                    timesteps, mean_rewards, std_rewards
                )

    print(f"Loaded {len(curves)} learning curves")
    return curves


# ══════════════════════════════════════════════════════════════════════════════
# Alignment helper
# ══════════════════════════════════════════════════════════════════════════════

def _align_and_stack(curves: dict, algo: str, condition: str):
    """
    Collects and aligns mean_reward arrays across seeds for (algo, condition).
    Trims ALL previously stored arrays when a shorter seed is encountered,
    preventing inhomogeneous array shape crash in np.array().
    Returns (timesteps, stacked_means) or (None, []) if no data.
    """
    all_timesteps = None
    all_means     = []

    for seed in SEEDS:
        key = (algo, condition, seed)
        if key not in curves:
            continue

        timesteps, mean_rewards, _ = curves[key]

        if all_timesteps is None:
            all_timesteps = timesteps.copy()
            all_means.append(mean_rewards.copy())
        else:
            min_len       = min(len(all_timesteps), len(mean_rewards))
            all_timesteps = all_timesteps[:min_len]
            all_means     = [m[:min_len] for m in all_means]
            all_means.append(mean_rewards[:min_len])

    if not all_means:
        return None, []

    return all_timesteps, np.array(all_means)


# ══════════════════════════════════════════════════════════════════════════════
# Learning curve plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_learning_curves_by_condition(
    curves:   dict,
    save_dir: str = "results/learning_curves",
):
    """One figure per algorithm showing all 3 curriculum conditions."""
    os.makedirs(save_dir, exist_ok=True)

    for algo in ALGORITHMS.keys():
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False

        for condition in CONDITIONS:
            ts, stacked = _align_and_stack(curves, algo, condition)
            if ts is None:
                continue

            mean  = stacked.mean(axis=0)
            std   = stacked.std(axis=0)
            color = CONDITION_COLORS[condition]
            label = CONDITION_LABELS[condition]

            ax.plot(ts, mean, color=color, label=label, linewidth=2)
            ax.fill_between(ts, mean - std, mean + std,
                            color=color, alpha=0.15)
            plotted = True

        if not plotted:
            plt.close()
            continue

        ax.axhline(y=0,   color="gray",  linewidth=0.8,
                   linestyle="--", alpha=0.5)
        ax.axhline(y=300, color="green", linewidth=0.8,
                   linestyle="--", alpha=0.5,
                   label="Solved threshold (300)")
        ax.set_xlabel("Timesteps", fontsize=13)
        ax.set_ylabel("Mean Evaluation Reward", fontsize=13)
        ax.set_title(
            f"{algo} — Learning Curves by Curriculum Condition",
            fontsize=14,
        )
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{algo}_by_condition.png")
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
        plt.close()


def plot_learning_curves_by_algorithm(
    curves:   dict,
    save_dir: str = "results/learning_curves",
):
    """One figure per curriculum condition showing all 4 algorithms."""
    os.makedirs(save_dir, exist_ok=True)

    for condition in CONDITIONS:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False

        for algo in ALGORITHMS.keys():
            ts, stacked = _align_and_stack(curves, algo, condition)
            if ts is None:
                continue

            mean  = stacked.mean(axis=0)
            std   = stacked.std(axis=0)
            color = ALGORITHM_COLORS[algo]

            ax.plot(ts, mean, color=color, label=algo, linewidth=2)
            ax.fill_between(ts, mean - std, mean + std,
                            color=color, alpha=0.15)
            plotted = True

        if not plotted:
            plt.close()
            continue

        ax.axhline(y=0,   color="gray",  linewidth=0.8,
                   linestyle="--", alpha=0.5)
        ax.axhline(y=300, color="green", linewidth=0.8,
                   linestyle="--", alpha=0.5,
                   label="Solved threshold (300)")
        ax.set_xlabel("Timesteps", fontsize=13)
        ax.set_ylabel("Mean Evaluation Reward", fontsize=13)
        ax.set_title(
            f"{CONDITION_LABELS[condition]} — "
            f"Learning Curves by Algorithm",
            fontsize=14,
        )
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(
            save_dir, f"{condition}_by_algorithm.png"
        )
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
        plt.close()


def plot_hardcore_learning_curves(
    curves:   dict,
    save_dir: str = "results/learning_curves",
):
    """One figure showing all algorithms trained directly on hardcore."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False

    for algo in ALGORITHMS.keys():
        ts, stacked = _align_and_stack(curves, algo, "hardcore")
        if ts is None:
            continue

        mean  = stacked.mean(axis=0)
        std   = stacked.std(axis=0)
        color = ALGORITHM_COLORS[algo]

        ax.plot(ts, mean, color=color, label=algo, linewidth=2)
        ax.fill_between(ts, mean - std, mean + std,
                        color=color, alpha=0.15)
        plotted = True

    if not plotted:
        print("No hardcore learning curves found — skipping.")
        plt.close()
        return

    ax.axhline(y=0, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.5)
    ax.set_xlabel("Timesteps", fontsize=13)
    ax.set_ylabel("Mean Evaluation Reward", fontsize=13)
    ax.set_title(
        "Hardcore Training — Learning Curves by Algorithm",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "hardcore_by_algorithm.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Results table
# ══════════════════════════════════════════════════════════════════════════════

def print_results_table(df: pd.DataFrame):
    """Prints summary table. ddof=0 avoids NaN when only one seed exists."""
    print("\n" + "="*65)
    print(
        f"  {'Algorithm':<10} {'Condition':<18} "
        f"{'Mean':>12} {'Std':>12}"
    )
    print("="*65)

    for algo in ALGORITHMS.keys():
        for condition in CONDITIONS:
            subset = df[
                (df["algorithm"] == algo) &
                (df["condition"] == condition)
            ]
            if subset.empty:
                continue
            mean = subset["mean_reward"].mean()
            std  = subset["mean_reward"].std(ddof=0)
            print(
                f"  {algo:<10} {condition:<18} "
                f"{mean:>12.2f} {std:>12.2f}"
            )
        print("-"*65)


# ══════════════════════════════════════════════════════════════════════════════
# Bar charts
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(
    df:        pd.DataFrame,
    save_path: str = "results/comparison_bar.png",
):
    """Grouped bar chart: algorithms × curriculum conditions."""
    os.makedirs("results", exist_ok=True)

    grouped = df.groupby(["algorithm", "condition"])["mean_reward"]
    means   = grouped.mean().unstack()
    stds    = grouped.std(ddof=0).unstack()

    valid = means.index[means.notna().any(axis=1)]
    means = means.loc[valid]
    stds  = stds.loc[valid]

    if means.empty:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x     = np.arange(len(means.index))
    width = 0.25

    for i, condition in enumerate(CONDITIONS):
        if condition not in means.columns:
            continue
        ax.bar(
            x + i * width,
            means[condition],
            width,
            yerr    = stds[condition].fillna(0),
            label   = CONDITION_LABELS[condition],
            color   = CONDITION_COLORS[condition],
            capsize = 4,
            alpha   = 0.85,
        )

    ax.axhline(y=0,   color="black", linewidth=0.8,
               linestyle="--", alpha=0.4)
    ax.axhline(y=300, color="green", linewidth=0.8,
               linestyle="--", alpha=0.5, label="Solved (300)")
    ax.set_xlabel("Algorithm", fontsize=13)
    ax.set_ylabel("Mean Reward (± std across seeds)", fontsize=13)
    ax.set_title(
        "BipedalWalker-v3: Final Performance by Algorithm and Curriculum",
        fontsize=14,
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(means.index, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def plot_hardcore_comparison(
    df_zero_shot: pd.DataFrame,
    df_trained:   pd.DataFrame = None,
    save_path:    str          = "results/hardcore_comparison.png",
):
    """Bar chart: zero-shot vs direct training on hardcore."""
    os.makedirs("results", exist_ok=True)

    algos       = list(ALGORITHMS.keys())
    x           = np.arange(len(algos))
    width       = 0.35
    has_trained = df_trained is not None and not df_trained.empty
    zs_offset   = -width / 2 if has_trained else 0.0

    zs_means, zs_stds = [], []
    for algo in algos:
        row = df_zero_shot[df_zero_shot["algorithm"] == algo]
        zs_means.append(
            float(row["hardcore_mean_reward"].iloc[0])
            if not row.empty else 0.0
        )
        zs_stds.append(
            float(row["hardcore_std_reward"].iloc[0])
            if not row.empty else 0.0
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x + zs_offset, zs_means, width, yerr=zs_stds,
        label   = "Zero-shot (best v3 model)",
        color   = "#607D8B", capsize=4, alpha=0.85,
    )

    if has_trained:
        tr_means, tr_stds = [], []
        for algo in algos:
            subset = df_trained[df_trained["algorithm"] == algo]
            tr_means.append(
                float(subset["mean_reward"].mean())
                if not subset.empty else 0.0
            )
            tr_stds.append(
                float(subset["mean_reward"].std(ddof=0))
                if not subset.empty else 0.0
            )
        ax.bar(
            x + width / 2, tr_means, width, yerr=tr_stds,
            label   = "Trained on Hardcore",
            color   = "#FF5722", capsize=4, alpha=0.85,
        )

    ax.axhline(y=0, color="black", linewidth=0.8,
               linestyle="--", alpha=0.4)
    ax.set_xlabel("Algorithm", fontsize=13)
    ax.set_ylabel("Mean Reward on Hardcore", fontsize=13)
    ax.set_title(
        "BipedalWalker Hardcore: Zero-shot vs Direct Training",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Master pipeline
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_results(
    model_base_dir:    str  = "models",
    eval_log_base_dir: str  = "eval_logs",
    n_eval_episodes:   int  = N_EVAL_EPISODES,
    use_best:          bool = True,
):
    """
    Runs the complete evaluation and plotting pipeline.
    Call once after all four teammates have finished training.

    use_best=True (default): loads best checkpoint — always recommended.
    Runs that solved the task early (reward >= 300) have identical
    best and final models. Runs that did not solve it may have a
    significantly better best than final due to policy collapse.
    """
    os.makedirs("results",                 exist_ok=True)
    os.makedirs("results/learning_curves", exist_ok=True)

    # Step 1: Primary evaluation
    print("\n" + "="*60)
    print(f"  STEP 1: Evaluating all primary models "
          f"[use_best={use_best}]")
    print("="*60)
    df_primary = evaluate_all(
        model_base_dir = model_base_dir,
        n_episodes     = n_eval_episodes,
        use_best       = use_best,
    )
    if not df_primary.empty:
        print_results_table(df_primary)
        plot_results(df_primary)
        df_primary.to_csv("results/primary_results.csv", index=False)
        print("Saved: results/primary_results.csv")

    # Step 2: Learning curves
    print("\n" + "="*60)
    print("  STEP 2: Plotting learning curves")
    print("="*60)
    curves = load_learning_curves(eval_log_base_dir=eval_log_base_dir)
    if curves:
        plot_learning_curves_by_condition(curves)
        plot_learning_curves_by_algorithm(curves)
        plot_hardcore_learning_curves(curves)

    # Step 3: Hardcore evaluation
    print("\n" + "="*60)
    print(f"  STEP 3: Hardcore evaluation [use_best={use_best}]")
    print("="*60)
    df_zero_shot = evaluate_zero_shot_hardcore(
        model_base_dir = model_base_dir,
        results_df     = df_primary if not df_primary.empty else None,
        n_episodes     = n_eval_episodes,
        use_best       = use_best,
    )
    df_trained = evaluate_hardcore_trained(
        model_base_dir = model_base_dir,
        n_episodes     = n_eval_episodes,
        use_best       = use_best,
    )

    if not df_zero_shot.empty:
        df_zero_shot.to_csv(
            "results/hardcore_zeroshot_results.csv", index=False
        )
        print("Saved: results/hardcore_zeroshot_results.csv")
    if not df_trained.empty:
        df_trained.to_csv(
            "results/hardcore_trained_results.csv", index=False
        )
        print("Saved: results/hardcore_trained_results.csv")
    if not df_zero_shot.empty:
        plot_hardcore_comparison(
            df_zero_shot,
            df_trained if not df_trained.empty else None,
        )

    print("\n" + "="*60)
    print("  ALL DONE")
    print("="*60)
    print("Results         → results/")
    print("Learning curves → results/learning_curves/")

    return df_primary, df_zero_shot, df_trained


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_all_results()