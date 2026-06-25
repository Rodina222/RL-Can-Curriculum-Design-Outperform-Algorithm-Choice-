"""
train.py

Single-model training for the BipedalWalker curriculum learning project.

Design decisions:
  - All four algorithms use rl-baselines3-zoo tuned hyperparameters
  - Lambda closure anti-pattern eliminated via named thunks throughout
  - Manual curriculum uses two independent callback sets (one per phase)
  - Early stopping when mean eval reward reaches 300 (solved threshold)
  - Eval environment is always standard BipedalWalker-v3, deterministic=True
  - SAC/TD3 train on GPU; PPO/A2C train on CPU (faster for MLP policies)
  - A2C uses VecNormalize (norm_obs + norm_reward) matching zoo normalize:true
  - SAC/TD3 replay buffer cleared at Phase 1→2 boundary (distribution shift)
  - A2C VecNormalize stats transferred Phase 1→Phase 2
  - VecNormalize stats saved BEFORE closing training env

Resumption design:
  - Single-phase: _find_latest_checkpoint() finds the latest .zip in
    checkpoints/. If found, loads it and trains for the remaining steps
    with reset_num_timesteps=False so the global step count is preserved.
  - Manual curriculum Phase 1: same checkpoint scan. After Phase 1
    finishes (normally or via EarlyStopping), phase1_model.zip is saved
    as an unambiguous completion marker.
  - Manual curriculum Phase 2: presence of phase1_model.zip signals Phase 1
    is done. _find_latest_checkpoint(after_steps=phase1_steps) finds any
    Phase 2 checkpoint. Phase 2 always runs for exactly
    (timesteps - MANUAL_TRANSFER_TIMESTEPS) additional steps regardless of
    when Phase 1 ended.

  NOTE — replay buffer on interruption:
    SB3 does NOT save the replay buffer inside checkpoint .zip files by
    default (it would add hundreds of MB per checkpoint). After loading a
    Phase 2 checkpoint the replay buffer is empty; SAC/TD3 will spend
    learning_starts (10K) steps collecting transitions before resuming
    gradient updates. This is a minor inefficiency, not a correctness issue.

  NOTE — early stopping and learning curve comparisons:
    StopTrainingOnRewardThreshold halts training when mean eval reward
    reaches 300. Runs that solve the task early will have fewer total
    timesteps than runs that do not. The report uses best checkpoint reward
    rather than final reward to ensure fair comparisons across runs with
    different stopping points. EvalCallback saves the best model
    automatically throughout training.

  NOTE — A2C VecNormalize:
    A2C uses VecNormalize (obs + reward normalisation) following the zoo
    configuration. All other algorithms use raw observations. This is not
    a confound — it reflects the correct training setup for each algorithm
    as established in the rl-baselines3-zoo literature. Disclosed in the
    report methodology section.

Execution flow (never run directly in production):
    imported by run_experiments.py → run_experiments.py called by teammates
"""

import os
import pickle
import torch
import numpy as np
import gymnasium as gym
from copy import deepcopy
from typing import Union

from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
)
from simplified_terrain import (
    make_env_no_curriculum,
    make_env_manual_curriculum,
    make_env_adaptive_curriculum,
)

# ── Constants ──────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS           = 1_000_000
MANUAL_TRANSFER_TIMESTEPS = 300_000
EVAL_FREQ                 = 10_000
EVAL_EPISODES             = 10
CHECKPOINT_FREQ           = 100_000
SOLVED_REWARD_THRESHOLD   = 300.0

ALGORITHMS = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "A2C": A2C,
}

# SAC and TD3 benefit from GPU (large replay buffer batch ops).
# PPO and A2C are faster on CPU for MLP policies.
GPU_ALGORITHMS = {"SAC", "TD3"}

# Zoo config for A2C BipedalWalker sets normalize:true.
# A2C is more sensitive to observation scale than off-policy methods
# because it has no replay buffer to smooth gradient variance.
NEEDS_VEC_NORMALIZE = {"A2C"}


# ── Learning rate / clip range schedules ──────────────────────────────────────

def _linear_schedule(initial_value: float):
    """
    Linear schedule: initial_value → 0 over training.
    SB3 passes progress_remaining in [1.0, 0.0] to schedule callables.
    """
    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return _schedule


# ── Tuned hyperparameters ──────────────────────────────────────────────────────
# Source: rl-baselines3-zoo BipedalWalker-v3 configurations.
# TD3 action_noise is NOT here — NormalActionNoise requires
# env.action_space.shape at runtime; injected in make_model().
HYPERPARAMS = {
    "PPO": dict(
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        learning_rate = _linear_schedule(2.5e-4),
        clip_range    = _linear_schedule(0.2),
        ent_coef      = 0.001,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        gae_lambda    = 0.95,
    ),
    "SAC": dict(
        learning_rate   = 7.3e-4,
        buffer_size     = 300_000,
        batch_size      = 256,
        gamma           = 0.98,
        tau             = 0.02,
        train_freq      = 8,
        gradient_steps  = 8,
        learning_starts = 10_000,
        ent_coef        = "auto",
        use_sde         = True,
        sde_sample_freq = 64,
    ),
    "TD3": dict(
        learning_rate       = _linear_schedule(1e-3),
        buffer_size         = 300_000,
        batch_size          = 256,
        gamma               = 0.98,
        tau                 = 0.005,
        train_freq          = 8,
        gradient_steps      = 8,
        policy_delay        = 2,
        target_policy_noise = 0.2,
        target_noise_clip   = 0.5,
        learning_starts     = 10_000,
        policy_kwargs       = dict(net_arch=[400, 300]),
        # action_noise: injected in make_model() — needs action_space.shape
    ),
    "A2C": dict(
        n_steps       = 8,
        gamma         = 0.99,
        gae_lambda    = 1.0,
        ent_coef      = 0.0,
        vf_coef       = 0.4,
        max_grad_norm = 0.5,
        learning_rate = _linear_schedule(9.6e-4),
        use_rms_prop  = False,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Resumption helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_latest_checkpoint(
    model_dir:   str,
    run_name:    str,
    after_steps: int = 0,
) -> tuple:
    """
    Finds the latest checkpoint saved by CheckpointCallback.

    SB3 CheckpointCallback names files:
        {run_name}_{timesteps}_steps.zip

    Args:
        model_dir:   Directory containing the checkpoints/ subdirectory.
        run_name:    The name_prefix used when creating CheckpointCallback.
        after_steps: Only return checkpoints with steps strictly greater than
                     this value. Used to distinguish Phase 2 checkpoints
                     (steps > phase1_steps) from Phase 1 checkpoints.

    Returns:
        (checkpoint_path, steps) or (None, 0) if no qualifying checkpoint.
    """
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        return None, 0

    best_path  = None
    best_steps = 0

    for fname in os.listdir(checkpoint_dir):
        if not (fname.startswith(run_name) and fname.endswith("_steps.zip")):
            continue
        try:
            # "PPO_manual_seed0_300000_steps.zip" → inner = "300000"
            inner = fname[len(run_name) + 1 : -len("_steps.zip")]
            steps = int(inner)
        except ValueError:
            continue

        if steps > best_steps and steps > after_steps:
            best_steps = steps
            best_path  = os.path.join(checkpoint_dir, fname)

    if best_path:
        print(
            f"  [Resume] Checkpoint found: {os.path.basename(best_path)} "
            f"({best_steps:,} steps)"
        )

    return best_path, best_steps


def _load_vecnorm_stats(model_dir: str) -> tuple:
    """
    Loads VecNormalize obs_rms and ret_rms from saved vecnormalize.pkl.

    Returns (obs_rms, ret_rms) or (None, None) if the file is absent or
    cannot be read.

    Used when resuming Phase 2 of the manual curriculum for A2C so that
    Phase 1 running normalisation statistics are transferred into the
    fresh Phase 2 environment instead of starting from zero.
    """
    vn_path = os.path.join(model_dir, "vecnormalize.pkl")
    if not os.path.exists(vn_path):
        return None, None

    try:
        with open(vn_path, "rb") as fh:
            vn = pickle.load(fh)
        return deepcopy(vn.obs_rms), deepcopy(vn.ret_rms)
    except Exception as exc:
        print(f"  [Warning] Could not load VecNormalize stats "
              f"from {vn_path}: {exc}")
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# Environment factories
# ══════════════════════════════════════════════════════════════════════════════

def _make_train_thunk(condition: str, seed: int, log_dir: str):
    """
    Returns a zero-argument thunk that builds one monitored training env.
    Named function captures parameters by value at definition time,
    eliminating the lambda closure bug.
    Hardcore seed is set inside the thunk so DummyVecEnv init cannot
    overwrite it.
    """
    def _thunk():
        os.makedirs(log_dir, exist_ok=True)

        if condition == "no_curriculum":
            env = make_env_no_curriculum(seed=seed)
        elif condition == "manual":
            env = make_env_manual_curriculum(seed=seed, roughness=0.2)
        elif condition == "adaptive":
            env = make_env_adaptive_curriculum(seed=seed)
        elif condition == "hardcore":
            env = gym.make("BipedalWalker-v3", hardcore=True)
            env.reset(seed=seed)
        else:
            raise ValueError(f"Unknown condition: '{condition}'")

        return Monitor(env, log_dir)

    return _thunk


def _make_std_v3_thunk(seed: int, log_dir: str = ""):
    """
    Thunk for seeded standard BipedalWalker-v3.

    seed+1 offset serves a specific scientific purpose: it ensures the
    Phase 2 terrain sequence is different from the Phase 1 terrain sequence
    for the same seed. Without this offset, a policy could appear to
    generalise to standard v3 simply by memorising the exact terrain layout
    it encountered during Phase 1, rather than having learned a truly
    transferable walking behaviour. The +1 offset makes Phase 2 terrain
    genuinely novel while keeping the run fully deterministic across
    restarts with the same seed.

    log_dir passed to Monitor so Phase 2 training episodes are logged.
    Used exclusively for Phase 2 of the manual curriculum.
    """
    def _thunk():
        env = gym.make("BipedalWalker-v3")
        env.reset(seed=seed + 1)
        return Monitor(env, log_dir) if log_dir else Monitor(env)

    return _thunk


def make_monitored_env(
    condition: str,
    seed:      int,
    log_dir:   str,
    algo_name: str = "",
) -> Union[DummyVecEnv, VecNormalize]:
    """
    Creates a single-worker vectorised training environment.
    A2C: wrapped with VecNormalize(norm_obs=True, norm_reward=True).
    """
    vec_env = DummyVecEnv([_make_train_thunk(condition, seed, log_dir)])

    if algo_name in NEEDS_VEC_NORMALIZE:
        vec_env = VecNormalize(
            vec_env,
            norm_obs    = True,
            norm_reward = True,
            clip_obs    = 10.0,
        )

    return vec_env


def _make_eval_env(algo_name: str = "") -> Union[DummyVecEnv, VecNormalize]:
    """
    Monitored eval environment on standard BipedalWalker-v3.
    Always standard v3 for fair cross-condition comparison.
    A2C: VecNormalize with training=False so eval does not corrupt stats.
    EvalCallback.sync_envs_normalization() syncs stats automatically.
    Seeding intentionally omitted: deterministic=True + 10 episodes
    averages out terrain randomness naturally.
    """
    def _thunk():
        return Monitor(gym.make("BipedalWalker-v3"))

    vec_env = DummyVecEnv([_thunk])

    if algo_name in NEEDS_VEC_NORMALIZE:
        vec_env = VecNormalize(
            vec_env,
            norm_obs    = True,
            norm_reward = False,
            clip_obs    = 10.0,
            training    = False,
        )

    return vec_env


# ══════════════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════════════

def make_model(
    algo_name: str,
    env:       Union[DummyVecEnv, VecNormalize],
    log_dir:   str,
    seed:      int,
    device:    str,
) -> Union[PPO, SAC, TD3, A2C]:
    """
    Initialises the algorithm with BipedalWalker-tuned hyperparameters.
    TD3 additionally receives NormalActionNoise(sigma=0.1) — cannot be
    stored in HYPERPARAMS because it requires env.action_space.shape.
    Without explicit action noise TD3 cannot explore and fails to learn.
    Only called for FRESH starts. Resumption uses ALGORITHMS[algo].load().
    """
    kwargs = dict(
        policy          = "MlpPolicy",
        env             = env,
        verbose         = 1,
        tensorboard_log = log_dir,
        seed            = seed,
        device          = device,
        **HYPERPARAMS[algo_name],
    )

    if algo_name == "TD3":
        n_actions          = env.action_space.shape[-1]
        kwargs["action_noise"] = NormalActionNoise(
            mean  = np.zeros(n_actions),
            sigma = 0.1 * np.ones(n_actions),
        )

    return ALGORITHMS[algo_name](**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Callback factory
# ══════════════════════════════════════════════════════════════════════════════

def _make_callbacks(
    eval_env:     Union[DummyVecEnv, VecNormalize],
    model_dir:    str,
    eval_log_dir: str,
    run_name:     str,
) -> list:
    """
    Fresh independent callback set for one training phase.
    Called separately per phase so Phase 2 EvalCallback starts with a
    clean best_mean_reward baseline, not Phase 1's inherited score.
    StopTrainingOnRewardThreshold halts at reward >= 300 (solved).
    Report uses best checkpoint reward for comparisons, not final reward,
    to handle runs with different stopping points fairly.
    """
    stop_on_solve = StopTrainingOnRewardThreshold(
        reward_threshold = SOLVED_REWARD_THRESHOLD,
        verbose          = 1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = f"{model_dir}/best",
        log_path             = eval_log_dir,
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = EVAL_EPISODES,
        deterministic        = True,
        render               = False,
        verbose              = 1,
        callback_on_new_best = stop_on_solve,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = CHECKPOINT_FREQ,
        save_path   = f"{model_dir}/checkpoints",
        name_prefix = run_name,
        verbose     = 1,
    )

    return [eval_callback, checkpoint_callback]


# ══════════════════════════════════════════════════════════════════════════════
# VecNormalize save helper
# ══════════════════════════════════════════════════════════════════════════════

def _save_vecnormalize(
    model:     Union[PPO, SAC, TD3, A2C],
    model_dir: str,
    algo_name: str,
) -> None:
    """
    Saves VecNormalize running stats to vecnormalize.pkl.
    No-op for all algorithms except A2C.
    MUST be called BEFORE closing the training environment.
    evaluate.py loads this file when evaluating A2C models.
    """
    if algo_name not in NEEDS_VEC_NORMALIZE:
        return

    vec_norm = model.get_vec_normalize_env()
    if vec_norm is not None:
        vn_path = os.path.join(model_dir, "vecnormalize.pkl")
        vec_norm.save(vn_path)
        print(f"✓ VecNormalize stats saved → {vn_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train(
    algo_name: str,
    condition: str,
    seed:      int,
    timesteps: int = TOTAL_TIMESTEPS,
) -> str:
    """
    Trains one model under one curriculum condition with one random seed.
    Automatically resumes from the latest checkpoint if the run was
    previously interrupted.

    Args:
        algo_name : One of PPO, SAC, TD3, A2C
        condition : One of no_curriculum, manual, adaptive, hardcore
        seed      : Integer random seed for reproducibility
        timesteps : Total number of environment steps

    Returns:
        Path to the saved final model (without .zip extension)

    Side effects (A2C only):
        Saves VecNormalize running stats to {model_dir}/vecnormalize.pkl.
        evaluate.py MUST load these stats when evaluating A2C models.

    Manual curriculum side effect:
        Saves {model_dir}/phase1_model.zip after Phase 1 completes.
        This file is the definitive completion marker used by resumption
        logic to decide whether Phase 1 needs to be re-run.
    """
    print(f"\n{'='*60}")
    print(f"  Algorithm : {algo_name}")
    print(f"  Condition : {condition}")
    print(f"  Seed      : {seed}")
    print(f"  Timesteps : {timesteps:,}")
    if algo_name in NEEDS_VEC_NORMALIZE:
        print(f"  VecNorm   : enabled (obs + reward)")
    print(f"{'='*60}\n")

    # ── Directories ────────────────────────────────────────────────────────────
    run_name     = f"{algo_name}_{condition}_seed{seed}"
    log_dir      = f"logs/{run_name}"
    model_dir    = f"models/{run_name}"
    eval_log_dir = f"eval_logs/{run_name}"
    os.makedirs(model_dir,    exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    # ── Device ─────────────────────────────────────────────────────────────────
    device = (
        "cuda"
        if torch.cuda.is_available() and algo_name in GPU_ALGORITHMS
        else "cpu"
    )
    print(f"  Device    : {device.upper()}")
    if device == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")

    eval_env = _make_eval_env(algo_name)

    # ══════════════════════════════════════════════════════════════════════════
    # Manual curriculum — two independent phases with resumption
    # ══════════════════════════════════════════════════════════════════════════
    if condition == "manual":

        # ── Phase 1 completion check ───────────────────────────────────────────
        phase1_marker   = os.path.join(model_dir, "phase1_model.zip")
        phase1_complete = os.path.exists(phase1_marker)

        # ── Phase 1 ────────────────────────────────────────────────────────────
        if not phase1_complete:
            ckpt_path, ckpt_steps = _find_latest_checkpoint(model_dir, run_name)
            p1_remaining          = max(0, MANUAL_TRANSFER_TIMESTEPS - ckpt_steps)

            print(
                f"[ManualCurriculum] Phase 1: Simplified terrain "
                f"({p1_remaining:,} of {MANUAL_TRANSFER_TIMESTEPS:,} "
                f"steps remaining)"
            )

            train_env_p1 = make_monitored_env("manual", seed, log_dir, algo_name)
            callbacks_p1 = _make_callbacks(eval_env, model_dir, eval_log_dir, run_name)

            if ckpt_path is not None and p1_remaining > 0:
                # Resume Phase 1 from checkpoint
                print(
                    f"  [Resume] Resuming Phase 1 from {ckpt_steps:,} steps "
                    f"({p1_remaining:,} remaining)"
                )
                model = ALGORITHMS[algo_name].load(
                    ckpt_path, env=train_env_p1, device=device
                )
                model.learn(
                    total_timesteps     = p1_remaining,
                    callback            = callbacks_p1,
                    tb_log_name         = run_name,
                    progress_bar        = True,
                    reset_num_timesteps = False,
                )

            elif ckpt_path is not None and p1_remaining == 0:
                # Checkpoint already at Phase 1 target
                print(
                    f"  [Resume] Phase 1 checkpoint already at target "
                    f"({ckpt_steps:,} steps). Skipping Phase 1 training."
                )
                model = ALGORITHMS[algo_name].load(
                    ckpt_path, env=train_env_p1, device=device
                )

            else:
                # Fresh Phase 1
                model = make_model(algo_name, train_env_p1, log_dir, seed, device)
                model.learn(
                    total_timesteps     = MANUAL_TRANSFER_TIMESTEPS,
                    callback            = callbacks_p1,
                    tb_log_name         = run_name,
                    progress_bar        = True,
                    reset_num_timesteps = True,
                )

            # Capture Phase 1 VecNorm stats before closing env
            p1_obs_rms = (
                deepcopy(train_env_p1.obs_rms)
                if isinstance(train_env_p1, VecNormalize) else None
            )
            p1_ret_rms = (
                deepcopy(train_env_p1.ret_rms)
                if isinstance(train_env_p1, VecNormalize) else None
            )

            # Save VecNorm stats BEFORE closing
            _save_vecnormalize(model, model_dir, algo_name)

            # Save phase1_model.zip — definitive Phase 1 completion marker
            p1_save      = os.path.join(model_dir, "phase1_model")
            model.save(p1_save)
            phase1_steps = int(model.num_timesteps)
            print(
                f"  [ManualCurriculum] Phase 1 complete at "
                f"{phase1_steps:,} steps → {p1_save}.zip"
            )

            train_env_p1.close()

        else:
            # Phase 1 already complete from a previous run
            print(
                f"[ManualCurriculum] Phase 1 already complete "
                f"(phase1_model.zip found). Skipping to Phase 2."
            )
            _tmp         = ALGORITHMS[algo_name].load(
                os.path.join(model_dir, "phase1_model"), device=device
            )
            phase1_steps = int(_tmp.num_timesteps)
            del _tmp
            print(f"  Phase 1 ended at {phase1_steps:,} steps")

            # Load Phase 1 VecNorm stats from disk (A2C only)
            p1_obs_rms, p1_ret_rms = _load_vecnorm_stats(model_dir)

        # ── Phase 2 ────────────────────────────────────────────────────────────
        p2_budget    = timesteps - MANUAL_TRANSFER_TIMESTEPS
        p2_target    = phase1_steps + p2_budget

        p2_ckpt_path, p2_ckpt_steps = _find_latest_checkpoint(
            model_dir, run_name, after_steps=phase1_steps
        )

        if p2_ckpt_path is not None:
            p2_start_path  = p2_ckpt_path
            p2_start_steps = p2_ckpt_steps
        else:
            p2_start_path  = os.path.join(model_dir, "phase1_model.zip")
            p2_start_steps = phase1_steps

        p2_remaining = max(0, p2_target - p2_start_steps)

        print(
            f"\n[ManualCurriculum] Phase 2: Standard v3 "
            f"({p2_remaining:,} remaining of {p2_budget:,} step budget)"
        )

        if p2_remaining == 0:
            print("[ManualCurriculum] Phase 2 already complete.")
            model = ALGORITHMS[algo_name].load(
                p2_start_path, env=eval_env, device=device
            )

        else:
            # log_dir passed to Monitor so Phase 2 episodes are logged
            base_env_p2 = DummyVecEnv([_make_std_v3_thunk(seed, log_dir)])

            if algo_name in NEEDS_VEC_NORMALIZE:
                train_env_p2 = VecNormalize(
                    base_env_p2,
                    norm_obs    = True,
                    norm_reward = True,
                    clip_obs    = 10.0,
                )
                if p1_obs_rms is not None:
                    train_env_p2.obs_rms = p1_obs_rms
                    train_env_p2.ret_rms = p1_ret_rms
                    print("[ManualCurriculum] VecNormalize stats transferred "
                          "Phase 1 → Phase 2.")
            else:
                train_env_p2 = base_env_p2

            model = ALGORITHMS[algo_name].load(
                p2_start_path, env=train_env_p2, device=device
            )

            # Clear replay buffer only when starting Phase 2 fresh.
            # Phase 1 transitions were on simplified terrain (different
            # observation distribution). SB3 does not save the replay buffer
            # in checkpoints by default so it is already empty on resume —
            # the explicit reset here documents intent and future-proofs
            # against SB3 adding save_replay_buffer=True.
            if p2_ckpt_path is None:
                if hasattr(model, "replay_buffer") and model.replay_buffer is not None:
                    model.replay_buffer.reset()
                    print(
                        "[ManualCurriculum] Replay buffer cleared "
                        "(off-policy: Phase 1→2 distribution boundary)."
                    )

            model.set_env(train_env_p2)
            callbacks_p2 = _make_callbacks(eval_env, model_dir, eval_log_dir, run_name)

            model.learn(
                total_timesteps     = p2_remaining,
                callback            = callbacks_p2,
                tb_log_name         = run_name,
                progress_bar        = True,
                reset_num_timesteps = False,
            )

            # Save VecNorm stats BEFORE closing Phase 2 env
            _save_vecnormalize(model, model_dir, algo_name)
            train_env_p2.close()

    # ══════════════════════════════════════════════════════════════════════════
    # Single-phase training with resumption
    # ══════════════════════════════════════════════════════════════════════════
    else:
        ckpt_path, ckpt_steps = _find_latest_checkpoint(model_dir, run_name)
        remaining             = max(0, timesteps - ckpt_steps)

        print(
            f"[{condition}] {remaining:,} steps remaining of {timesteps:,}"
        )

        train_env = make_monitored_env(condition, seed, log_dir, algo_name)
        callbacks = _make_callbacks(eval_env, model_dir, eval_log_dir, run_name)

        if ckpt_path is not None and remaining > 0:
            # Resume from checkpoint
            print(
                f"  [Resume] Resuming from {ckpt_steps:,} steps "
                f"({remaining:,} remaining)"
            )
            model = ALGORITHMS[algo_name].load(
                ckpt_path, env=train_env, device=device
            )
            model.learn(
                total_timesteps     = remaining,
                callback            = callbacks,
                tb_log_name         = run_name,
                progress_bar        = True,
                reset_num_timesteps = False,
            )

        elif ckpt_path is not None and remaining == 0:
            # Checkpoint already at target
            print(
                f"  [Resume] Checkpoint already at target "
                f"({ckpt_steps:,} steps). Saving as final model."
            )
            model = ALGORITHMS[algo_name].load(
                ckpt_path, env=train_env, device=device
            )

        else:
            # Fresh start
            model = make_model(algo_name, train_env, log_dir, seed, device)
            model.learn(
                total_timesteps = timesteps,
                callback        = callbacks,
                tb_log_name     = run_name,
                progress_bar    = True,
            )

        # Save VecNorm stats BEFORE closing training env
        _save_vecnormalize(model, model_dir, algo_name)
        train_env.close()

    # ── Save final model ───────────────────────────────────────────────────────
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"\n✓ Model saved → {final_path}.zip")

    eval_env.close()
    return final_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point (for debugging a single run only)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description = "Train a single RL algorithm on BipedalWalker-v3. "
                      "For batch experiments use run_experiments.py instead."
    )
    parser.add_argument(
        "--algo",
        type    = str,
        default = "PPO",
        choices = list(ALGORITHMS.keys()),
        help    = "RL algorithm to train (default: PPO)",
    )
    parser.add_argument(
        "--condition",
        type    = str,
        default = "no_curriculum",
        choices = ["no_curriculum", "manual", "adaptive", "hardcore"],
        help    = "Curriculum condition (default: no_curriculum)",
    )
    parser.add_argument(
        "--seed",
        type    = int,
        default = 0,
        help    = "Random seed (default: 0)",
    )
    parser.add_argument(
        "--timesteps",
        type    = int,
        default = TOTAL_TIMESTEPS,
        help    = f"Total training timesteps (default: {TOTAL_TIMESTEPS:,})",
    )
    args = parser.parse_args()

    train(
        algo_name = args.algo,
        condition = args.condition,
        seed      = args.seed,
        timesteps = args.timesteps,
    )