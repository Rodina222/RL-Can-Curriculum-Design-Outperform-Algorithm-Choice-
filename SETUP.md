# Setup Instructions

These instructions are based on actual debugging done on this project — every
step below exists because skipping it caused a confirmed failure during
setup. Read the "why" notes if something goes wrong; they're not filler.

---

## 1. Use Python 3.12 — explicitly, not whatever `python` defaults to

**This is the most important step.** `Box2D==2.3.10` (a dependency of
`gymnasium[box2d]`) only has prebuilt wheels for Python 3.10, 3.11, and 3.12.
There is **no wheel for Python 3.13 or 3.14**. If your machine's default
`python` resolves to 3.14 (common on a fresh Windows install), pip cannot
find a Box2D wheel and the install fails with no clear fix — pip will try to
compile from source instead, which requires Visual Studio build tools and
SWIG configured correctly, and is not something teammates should have to set
up individually.

Check what's installed:

```powershell
py -0
```

You need `3.12` in that list. If it's missing, install Python 3.12 from
[python.org](https://www.python.org/downloads/) before continuing.

Create the virtual environment using 3.12 **explicitly**:

```powershell
py -3.12 -m venv rl_env
.\rl_env\Scripts\Activate.ps1
python --version
```

The last command must print `Python 3.12.x`. If it prints anything else,
delete the `rl_env` folder and repeat this step — do not proceed with the
wrong interpreter.

> If `Activate.ps1` is blocked by execution policy:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> or use `.\rl_env\Scripts\activate.bat` instead, which has no policy restriction.

---

## 2. Install core dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This should complete using only prebuilt `.whl` files — you should **not**
see SWIG output or a C++ compiler running. If you do, stop: it means pip
fell back to building Box2D from source, which means Python 3.12 isn't
actually active in this venv. Go back to step 1.

`requirements.txt` deliberately does **not** include `torch`,
`torchvision`, or `torchaudio` — see the next step for why.

---

## 3. Install PyTorch separately (GPU build)

The GPU/CUDA build of PyTorch is **not available on PyPI** — it's hosted on
PyTorch's own package index, organized by CUDA version tag. A plain
`pip install -r requirements.txt` can never resolve a `+cuXXX`-tagged torch
build, regardless of Python version. This is why torch is installed as a
separate command:

```powershell
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

**Why torch 2.8.0 specifically, not an older version:** `stable-baselines3`
requires `torch>=2.8`. CUDA 12.1-tagged torch builds only go up to torch
2.5.x, so a `cu121` tag cannot satisfy that requirement — pip will report
`ResolutionImpossible` if you try. CUDA 12.8 (`cu128`) builds support
`torch>=2.8`, which is why this project pairs `cu128` with `torch==2.8.0`,
not the more commonly-seen `cu121`/`torch==2.5.1` combination.

**If this machine has no NVIDIA GPU:**

```powershell
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

(omit `--index-url`; this installs the CPU-only build from PyPI directly)

**If this machine has a different CUDA version:** check
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
for the correct `--index-url` for your CUDA toolkit version, keeping
`torch>=2.8` to remain compatible with `stable-baselines3`.

---

## 4. Automated install (optional)

Steps 2–3 are combined in `install.ps1`. After completing step 1:

```powershell
.\install.ps1
```

This runs `pip install -r requirements.txt`, then installs the CUDA 12.8
torch build, then verifies everything imports correctly (step 5 below).

---

## 5. Verify the installation

Run these checks before training anything:

```powershell
python -c "import Box2D; print('Box2D OK')"
python -c "import gymnasium as gym; gym.make('BipedalWalker-v3'); print('BipedalWalker OK')"
python -c "import stable_baselines3; print('stable_baselines3 OK')"
python -c "import torch; print('torch OK, CUDA available:', torch.cuda.is_available())"
```

All four must print `OK` with no errors. The last line should print
`CUDA available: True` on a machine with a working NVIDIA GPU and up-to-date
drivers. If it prints `False` unexpectedly on a GPU machine, recheck step 3 —
torch likely installed the CPU build by accident.

---

## 6. Run a single test before the full batch

Before committing to the full 32-run experiment sweep, confirm the pipeline
actually trains end-to-end with a short run:

```powershell
python train.py --algo PPO --condition no_curriculum --seed 0 --timesteps 10000
```

If this completes and saves a model under `models/PPO_no_curriculum_seed0/`,
the environment is correctly set up and you're ready to run the full batch
described in [README.md](README.md).

---

## Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `No matching distribution found for Box2D` | Wrong Python version active (3.13/3.14) | Recreate venv with `py -3.12 -m venv rl_env` |
| SWIG/C++ compiler output during `pip install` | Same as above — pip fell back to source build | Same fix |
| `No matching distribution found for torch==X+cuYYY` | Tried to install CUDA torch via `requirements.txt`/default index | Install torch separately per step 3 |
| `ResolutionImpossible` mentioning torch and stable-baselines3 | CUDA tag too old for required torch version (e.g. cu121 capped at torch 2.5.x, but stable-baselines3 needs ≥2.8) | Use `cu128` + `torch==2.8.0` as in step 3 |
| `ModuleNotFoundError: No module named '_distutils_hack'` when activating venv | Corrupted/inconsistent setuptools in that specific venv | Delete the venv folder and recreate from scratch with `py -3.12` |
| `pip show stable_baselines3` reports "not found" after install appeared to succeed | An earlier step in the same `pip install` run failed (e.g. the torch conflict above), so **nothing** in that command was actually installed | Check the full install log for an `ERROR:` line, fix it, and re-run the full install command |
