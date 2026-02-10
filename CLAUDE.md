# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical prediction web app for TAVR (Transcatheter Aortic Valve Replacement) in-hospital mortality risk. Uses a pre-trained PyCaret logistic regression model (L2 penalty) trained on HCUP NIS data (2012-2019). Published in *Scientific Reports*: Alhwiti, T., Aldrugh, S., & Megahed, F. M. (2023).

**Primary repo:** GitHub. Auto-syncs to HF Spaces on every push to `main`.
**Deployed at:** `https://huggingface.co/spaces/fmegahed/tavr_project` (Docker-based Space)

## Running the App

```bash
# Local
python app.py

# Docker
docker build -t tavr-app .
docker run -p 7860:7860 tavr-app
```

The app launches on port 7860 (configurable via `PORT` or `GRADIO_SERVER_PORT` env vars).

## Architecture

Single-file application (`app.py`) with three layers:

1. **Model loading** (`_ensure_model_file()` / `_get_model()`): Lazy-loads `final_model.pkl` with global caching. Falls back to downloading from GitHub if the local file is missing.

2. **Prediction** (`predict()`): Takes 44 patient parameters (demographics, hospital info, 32 comorbidities), constructs a DataFrame with proper categorical types (including ordered categoricals for `zipinc_qrtl`), runs PyCaret's `predict_model()` with `raw_score=True`. Returns `{"Death": 0-1, "Survival": 0-1}` dict for `gr.Label`.

3. **Gradio UI** (`gr.Blocks` with `gr.themes.Soft()`): 44 inputs organized in 4 tabs (Patient Demographics, Hospital Information, Comorbidities, Procedure). Explicit "Predict" button triggers prediction; output displayed as `gr.Label` with confidence bars.

## GitHub Actions Workflows

- **`sync-to-hf.yml`**: Pushes to HF Spaces on every `main` push. Requires `HF_TOKEN` secret (write-access).
- **`daily-keepalive.yml`**: Runs daily at 06:00 UTC. Updates `keep_alive.csv` with current date/timestamp and pushes to both GitHub and HF to prevent Space from sleeping.

## Key Constraints

- **Python 3.8** — pinned in the Dockerfile, all dependencies must be compatible
- **PyCaret 2.3.6** — requires `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True` env var for installation
- **All dependency versions are pinned** in `requirements.txt` — changing one may break others due to tight PyCaret/scikit-learn compatibility
- **`final_model.pkl`** is tracked via Git LFS (see `.gitattributes`)
- The model expects exact categorical column names and types; any change to input names in `predict()` will break predictions
- Gradio version is 3.50.2 (not 4.x) — API differs significantly from current Gradio

## GitHub Setup

1. Create a GitHub repo and add it as the `origin` remote
2. Add `HF_TOKEN` secret (Settings > Secrets > Actions) with a write-access token from https://huggingface.co/settings/tokens
3. Push to `main` — the sync workflow will mirror to HF Spaces automatically
