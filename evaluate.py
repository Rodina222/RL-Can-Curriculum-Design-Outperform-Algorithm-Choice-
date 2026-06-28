"""
evaluate.py — Updated with 5 Professional Metrics

Metrics:
1. Mean Reward ± Std        (primary)
2. Success Rate             (primary)
3. Sample Efficiency        (primary)
4. Seed Sensitivity         (secondary)
5. Zero-Shot Hardcore       (secondary — PPO + SAC only)
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
SUCCESS_THRESHOLD = 200.0    # reward > 200 = robot completed the course
SAMPLE_EFF_THRESHOLD = 100.0 # reward > 100 = robot learned basic walking

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
# Single model evaluation — returns 3 metrics
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    algo_name:  str,
    run_dir:    str,
    env_id:     str  = "BipedalWalker-v3",
    hardcore:   bool = False,
    n_episodes: int  = N_EVAL_EPISODES,
    render:     bool = False,
    use_best:   bool = True,
) -> tuple:
    """
    Evaluates a trained model.

    Returns:
        (mean_reward, std_reward, success_rate)

        mean_reward:  Average total reward across n_episodes
        std_reward:   Standard deviation of episode rewards
        success_rate: % of episodes with reward > SUCCESS_THRESHOLD (200)

    Why success_rate?
        Mean reward alone is misleading — a robot walking halfway
        consistently vs one that occasionally finishes can have the
        same mean reward but represent fundamentally different behaviors.
    """

    # ── Resolve model path ────────────────────────────────────────────────────
    best_path  = os.path.join(run_dir, "best", "best_model")
    final_path = os.path.join(run_dir, "final_model")

    if use_best and os.path.exists(best_path + ".zip"):
        model_path = best_path
        print(f"  [✓ Best ] {best_path}.zip")
    elif os.path.exists(final_path + ".zip"):
        if use_best:
            print(
                f"  [! Warn ] best_model.zip not found in {run_dir}/best/\n"
                f"            Falling back to final_model.zip."
            )
        model_path = final_path
        print(f"  [  Final] {final_path}.zip")
    else:
        raise FileNotFoundError(
            f"No model found in {run_dir}.\n"
            f"Looked for:\n  {best_path}.zip\n  {final_path}.zip"
        )

    # ── Build environment ─────────────────────────────────────────────────────
    render_mode = "human" if render else None
    tag = " [HC]" if hardcore else ""

    if hardcore:
        raw_env = gym.make("BipedalWalker-v3", hardcore=True,
                           render_mode=render_mode)
    else:
        raw_env = gym.make(env_id, render_mode=render_mode)

    algo_class = ALGORITHMS[algo_name]
    model = algo_class.load(model_path)

    # ── A2C: VecNormalize required ────────────────────────────────────────────
    if algo_name == "A2C":
        vn_path = os.path.join(run_dir, "vecnormalize.pkl")
        if not os.path.exists(vn_path):
            raise FileNotFoundError(
                f"vecnormalize.pkl not found at {vn_path}.\n"
                f"A2C requires normalisation stats from training."
            )

        def _make_a2c_eval_thunk(env):
            def _thunk():
                return Monitor(env)
            return _thunk

        vec_env          = DummyVecEnv([_make_a2c_eval_thunk(raw_env)])
        vec_env          = VecNormalize.load(vn_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        episode_rewards = []
        for episode in range(n_episodes):
            obs   = vec_env.reset()
            done  = False
            total = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done_arr, _ = vec_env.step(action)
                done  = bool(done_arr[0])
                total += float(reward[0])
            episode_rewards.append(total)
            print(f"  Episode {episode+1:02d}/{n_episodes}{tag}: {total:.2f}")

        vec_env.close()

    else:
        # ── All other algorithms ──────────────────────────────────────────────
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

    # ── Compute metrics ───────────────────────────────────────────────────────
    mean_reward  = float(np.mean(episode_rewards))
    std_reward   = float(np.std(episode_rewards))

    # Metric 2 — Success Rate
    # Why 200? BipedalWalker reward > 200 means robot walked most of the course
    success_rate = float(
        len([r for r in episode_rewards if r > SUCCESS_THRESHOLD])
        / len(episode_rewards) * 100
    )

    print(f"  → Mean: {mean_reward:.2f} ± {std_reward:.2f} | "
          f"Success: {success_rate:.0f}%")

    return mean_reward, std_reward, success_rate


# ══════════════════════════════════════════════════════════════════════════════
# Metric 3 — Sample Efficiency from learning curves
# ══════════════════════════════════════════════════════════════════════════════

def compute_sample_efficiency(curves: dict) -> dict:
    """
    Computes sample efficiency for each (algo, condition) pair.

    Sample efficiency = timestep at which mean reward first exceeds
    SAMPLE_EFF_THRESHOLD (100) averaged across seeds.

    Why reward > 100?
        Score of 100 means robot is walking consistently without falling.
        Below 100 = still stumbling. Above 100 = basic locomotion achieved.

    Returns:
        dict keyed by (algo, condition) → mean timestep (or None if never)
    """
    efficiency = {}

    for algo in ALGORITHMS.keys():
        for condition in CONDITIONS:
            steps_list = []

            for seed in SEEDS:
                key = (algo, condition, seed)
                if key not in curves:
                    continue

                timesteps, mean_rewards, _ = curves[key]

                # Find first timestep where reward exceeds threshold
                above = np.where(mean_rewards >= SAMPLE_EFF_THRESHOLD)[0]
                if len(above) > 0:
                    steps_list.append(int(timesteps[above[0]]))
                else:
                    steps_list.append(None)  # Never reached threshold

            if steps_list:
                valid = [s for s in steps_list if s is not None]
                efficiency[(algo, condition)] = (
                    int(np.mean(valid)) if valid else None
                )

    return efficiency


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
    Evaluates all trained primary models.
    Returns DataFrame with mean_reward, std_reward, success_rate per run.
    """
    results = []

    for algo in ALGORITHMS.keys():
        for condition in CONDITIONS:
            for seed in SEEDS:
                run_name = f"{algo}_{condition}_seed{seed}"
                run_dir  = os.path.join(model_base_dir, run_name)

                if not os.path.exists(
                    os.path.join(run_dir, "final_model.zip")
                ):
                    print(f"[SKIP] Run not complete: {run_name}")
                    continue

                print(f"\nEvaluating: {algo} | {condition} | seed {seed}")

                try:
                    mean, std, success = evaluate_model(
                        algo, run_dir,
                        env_id     = env_id,
                        n_episodes = n_episodes,
                        use_best   = use_best,
                    )
                except FileNotFoundError as e:
                    print(f"  [ERROR] {e}")
                    continue

                results.append({
                    "algorithm":    algo,
                    "condition":    condition,
                    "seed":         seed,
                    "mean_reward":  mean,
                    "std_reward":   std,
                    "success_rate": success,
                })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# Metric 4 — Seed Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def compute_seed_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes seed sensitivity for each (algo, condition) pair.

    Seed sensitivity = |reward_seed0 - reward_seed1|

    Why it matters:
        High sensitivity means the result depends heavily on random
        initialization — not reliable. Low sensitivity = robust algorithm.

    Returns DataFrame with seed_sensitivity column added.
    """
    summary_rows = []

    for algo in ALGORITHMS.keys():
        for condition in CONDITIONS:
            subset = df[
                (df["algorithm"] == algo) &
                (df["condition"] == condition)
            ]
            if subset.empty:
                continue

            mean_reward  = subset["mean_reward"].mean()
            std_reward   = subset["std_reward"].mean()
            success_rate = subset["success_rate"].mean()

            # Metric 4 — Seed Sensitivity
            if len(subset) == 2:
                rewards = subset["mean_reward"].values
                seed_sensitivity = abs(rewards[0] - rewards[1])
            else:
                seed_sensitivity = None

            summary_rows.append({
                "algorithm":        algo,
                "condition":        condition,
                "mean_reward":      round(mean_reward, 2),
                "std_reward":       round(std_reward, 2),
                "success_rate":     round(success_rate, 1),
                "seed_sensitivity": round(seed_sensitivity, 2)
                                    if seed_sensitivity is not None else None,
            })

    return pd.DataFrame(summary_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Hardcore evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_hardcore_trained(
    model_base_dir: str  = "models",
    n_episodes:     int  = N_EVAL_EPISODES,
    use_best:       bool = True,
) -> pd.DataFrame:
    """Evaluates models trained directly on hardcore."""
    print("\n" + "="*60)
    print("  HARDCORE TRAINED MODELS EVALUATION")
    print("="*60)

    results = []

    for algo in ALGORITHMS.keys():
        for seed in SEEDS:
            run_name = f"{algo}_hardcore_seed{seed}"
            run_dir  = os.path.join(model_base_dir, run_name)

            if not os.path.exists(
                os.path.join(run_dir, "final_model.zip")
            ):
                print(f"[SKIP] Run not complete: {run_name}")
                continue

            print(f"\nEvaluating hardcore-trained: {algo} | seed {seed}")

            try:
                mean, std, success = evaluate_model(
                    algo, run_dir,
                    hardcore   = True,
                    n_episodes = n_episodes,
                    use_best   = use_best,
                )
            except FileNotFoundError as e:
                print(f"  [ERROR] {e}")
                continue

            results.append({
                "algorithm":    algo,
                "seed":         seed,
                "mean_reward":  mean,
                "std_reward":   std,
                "success_rate": success,
            })

    return pd.DataFrame(results)


def evaluate_zero_shot_hardcore(
    model_base_dir: str          = "models",
    results_df:     pd.DataFrame = None,
    n_episodes:     int          = N_EVAL_EPISODES,
    use_best:       bool         = True,
) -> pd.DataFrame:
    """
    Evaluates best v3-trained model per algorithm zero-shot on Hardcore.

    Metric 5 — Zero-Shot Transfer Ratio:
        transfer_ratio = hardcore_reward / v3_reward × 100
        100% = perfect transfer
        0%   = complete failure

    Note: Only meaningful for PPO and SAC.
          A2C failed on v3 (0% success) — Hardcore result uninformative.
          TD3 seed-sensitive — note in report.
    """
    print("\n" + "="*60)
    print("  ZERO-SHOT HARDCORE EVALUATION")
    print("="*60)

    results = []

    for algo in ALGORITHMS.keys():

        # Find best v3 model
        if results_df is not None and not results_df.empty:
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
            print(f"[SKIP] No results_df provided for {algo}")
            continue

        if not os.path.exists(os.path.join(run_dir, "final_model.zip")):
            print(f"[SKIP] Model files missing for {algo}")
            continue

        print(
            f"\n{algo} best v3: {best_condition} seed {best_seed} "
            f"(v3 mean: {best_v3_mean:.2f})"
        )
        print("Testing zero-shot on Hardcore...")

        try:
            mean_hc, std_hc, success_hc = evaluate_model(
                algo, run_dir,
                hardcore   = True,
                n_episodes = n_episodes,
                use_best   = use_best,
            )
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            continue

        # Metric 5 — Transfer Ratio
        # How much of v3 performance transferred to Hardcore?
        transfer_ratio = (
            (mean_hc / best_v3_mean * 100)
            if best_v3_mean > 0 else 0.0
        )

        print(f"  Zero-shot hardcore: {mean_hc:.2f} ± {std_hc:.2f}")
        print(f"  Transfer ratio: {transfer_ratio:.1f}%")

        results.append({
            "algorithm":            algo,
            "best_v3_condition":    best_condition,
            "best_v3_seed":         best_seed,
            "v3_mean_reward":       best_v3_mean,
            "hardcore_mean_reward": mean_hc,
            "hardcore_std_reward":  std_hc,
            "hardcore_success_rate": success_hc,
            "transfer_ratio":       round(transfer_ratio, 1),
        })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# Learning curves
# ══════════════════════════════════════════════════════════════════════════════

def load_learning_curves(eval_log_base_dir: str = "eval_logs") -> dict:
    """Loads EvalCallback evaluations.npz files."""
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


def _align_and_stack(curves: dict, algo: str, condition: str):
    """Aligns and stacks mean_reward arrays across seeds."""
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

        # Sample efficiency threshold line
        ax.axhline(y=SAMPLE_EFF_THRESHOLD, color="orange",
                   linewidth=1.5, linestyle=":",
                   label=f"Sample efficiency threshold ({SAMPLE_EFF_THRESHOLD})")
        ax.axhline(y=SUCCESS_THRESHOLD, color="blue",
                   linewidth=1.5, linestyle=":",
                   label=f"Success threshold ({SUCCESS_THRESHOLD})")
        ax.axhline(y=300, color="green", linewidth=0.8,
                   linestyle="--", alpha=0.5, label="Solved (300)")

        ax.set_xlabel("Timesteps", fontsize=13)
        ax.set_ylabel("Mean Evaluation Reward", fontsize=13)
        ax.set_title(
            f"{algo} — Learning Curves by Curriculum Condition",
            fontsize=14,
        )
        ax.legend(fontsize=10)
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

        ax.axhline(y=SAMPLE_EFF_THRESHOLD, color="orange",
                   linewidth=1.5, linestyle=":",
                   label=f"Sample efficiency threshold ({SAMPLE_EFF_THRESHOLD})")
        ax.axhline(y=300, color="green", linewidth=0.8,
                   linestyle="--", alpha=0.5, label="Solved (300)")

        ax.set_xlabel("Timesteps", fontsize=13)
        ax.set_ylabel("Mean Evaluation Reward", fontsize=13)
        ax.set_title(
            f"{CONDITION_LABELS[condition]} — Learning Curves by Algorithm",
            fontsize=14,
        )
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{condition}_by_algorithm.png")
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
        ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.15)
        plotted = True

    if not plotted:
        print("No hardcore learning curves found — skipping.")
        plt.close()
        return

    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Timesteps", fontsize=13)
    ax.set_ylabel("Mean Evaluation Reward", fontsize=13)
    ax.set_title("Hardcore Training — Learning Curves by Algorithm", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "hardcore_by_algorithm.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Results table — updated with 5 metrics
# ══════════════════════════════════════════════════════════════════════════════

def print_results_table(df_summary: pd.DataFrame, efficiency: dict = None):
    """
    Prints professional results table with all 5 metrics.
    """
    print("\n" + "="*85)
    print(
        f"  {'Algorithm':<8} {'Condition':<18} "
        f"{'Mean Reward':>12} {'Std':>8} "
        f"{'Success%':>10} {'Sensitivity':>13} {'Sample Eff.':>12}"
    )
    print("="*85)

    for algo in ALGORITHMS.keys():
        for condition in CONDITIONS:
            subset = df_summary[
                (df_summary["algorithm"] == algo) &
                (df_summary["condition"] == condition)
            ]
            if subset.empty:
                continue

            row = subset.iloc[0]
            eff = efficiency.get((algo, condition)) if efficiency else None
            eff_str = f"{eff:,}" if eff else "Never"

            print(
                f"  {algo:<8} {condition:<18} "
                f"{row['mean_reward']:>12.1f} "
                f"{row['std_reward']:>8.1f} "
                f"{row['success_rate']:>9.0f}% "
                f"{row['seed_sensitivity'] if row['seed_sensitivity'] else 'N/A':>13} "
                f"{eff_str:>12}"
            )
        print("-"*85)


# ══════════════════════════════════════════════════════════════════════════════
# Bar charts
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(
    df:        pd.DataFrame,
    save_path: str = "results/comparison_bar.png",
):
    """Grouped bar chart: mean reward by algorithm × curriculum condition."""
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
    ax.axhline(y=200, color="blue",  linewidth=1.0,
               linestyle=":",  alpha=0.6, label="Success threshold (200)")
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


def plot_success_rate(
    df:        pd.DataFrame,
    save_path: str = "results/success_rate_bar.png",
):
    """Grouped bar chart: success rate by algorithm × curriculum condition."""
    os.makedirs("results", exist_ok=True)

    grouped = df.groupby(["algorithm", "condition"])["success_rate"]
    means   = grouped.mean().unstack()

    valid = means.index[means.notna().any(axis=1)]
    means = means.loc[valid]

    if means.empty:
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
            label = CONDITION_LABELS[condition],
            color = CONDITION_COLORS[condition],
            alpha = 0.85,
        )

    ax.set_xlabel("Algorithm", fontsize=13)
    ax.set_ylabel("Success Rate (%)", fontsize=13)
    ax.set_title(
        "BipedalWalker-v3: Success Rate by Algorithm and Curriculum",
        fontsize=14,
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(means.index, fontsize=12)
    ax.set_ylim(0, 115)
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
        zs_means.append(float(row["hardcore_mean_reward"].iloc[0])
                        if not row.empty else 0.0)
        zs_stds.append(float(row["hardcore_std_reward"].iloc[0])
                       if not row.empty else 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x + zs_offset, zs_means, width, yerr=zs_stds,
           label="Zero-shot (best v3 model)",
           color="#607D8B", capsize=4, alpha=0.85)

    if has_trained:
        tr_means, tr_stds = [], []
        for algo in algos:
            subset = df_trained[df_trained["algorithm"] == algo]
            tr_means.append(float(subset["mean_reward"].mean())
                            if not subset.empty else 0.0)
            tr_stds.append(float(subset["mean_reward"].std(ddof=0))
                           if not subset.empty else 0.0)
        ax.bar(x + width / 2, tr_means, width, yerr=tr_stds,
               label="Trained on Hardcore",
               color="#FF5722", capsize=4, alpha=0.85)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("Algorithm", fontsize=13)
    ax.set_ylabel("Mean Reward on Hardcore", fontsize=13)
    ax.set_title("Hardcore: Zero-shot Transfer vs Direct Training", fontsize=14)
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
    Complete evaluation pipeline with all 5 metrics.

    Outputs:
        results/primary_results.csv          ← raw per-seed results
        results/summary_results.csv          ← aggregated with all 5 metrics
        results/hardcore_zeroshot_results.csv
        results/hardcore_trained_results.csv
        results/comparison_bar.png
        results/success_rate_bar.png
        results/hardcore_comparison.png
        results/learning_curves/
    """
    os.makedirs("results",                 exist_ok=True)
    os.makedirs("results/learning_curves", exist_ok=True)

    # ── Step 1: Primary evaluation ────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  STEP 1: Evaluating all primary models")
    print("="*60)
    df_primary = evaluate_all(
        model_base_dir = model_base_dir,
        n_episodes     = n_eval_episodes,
        use_best       = use_best,
    )

    if not df_primary.empty:
        df_primary.to_csv("results/primary_results.csv", index=False)
        print("Saved: results/primary_results.csv")

        # Metric 4 — Seed Sensitivity summary
        df_summary = compute_seed_sensitivity(df_primary)

        # ── Step 2: Learning curves + Sample Efficiency ───────────────────────
        print("\n" + "="*60)
        print("  STEP 2: Learning curves + Sample Efficiency")
        print("="*60)
        curves = load_learning_curves(eval_log_base_dir=eval_log_base_dir)

        # Metric 3 — Sample Efficiency
        efficiency = {}
        if curves:
            efficiency = compute_sample_efficiency(curves)
            plot_learning_curves_by_condition(curves)
            plot_learning_curves_by_algorithm(curves)
            plot_hardcore_learning_curves(curves)

        # Print full results table with all 5 metrics
        print_results_table(df_summary, efficiency)

        # Add sample efficiency to summary CSV
        df_summary["sample_efficiency"] = df_summary.apply(
            lambda r: efficiency.get((r["algorithm"], r["condition"])),
            axis=1
        )
        df_summary.to_csv("results/summary_results.csv", index=False)
        print("Saved: results/summary_results.csv")

        # Figures
        plot_results(df_primary)
        plot_success_rate(df_primary)

    # ── Step 3: Hardcore evaluation ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3: Hardcore evaluation")
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
        print("\nZero-shot transfer ratios:")
        for _, row in df_zero_shot.iterrows():
            print(f"  {row['algorithm']}: {row['transfer_ratio']}% transfer")

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
    print("Primary results  → results/primary_results.csv")
    print("Summary (5 metrics) → results/summary_results.csv")
    print("Learning curves  → results/learning_curves/")

    return df_primary, df_zero_shot, df_trained


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_all_results()