# Face Synthesis (Diffusion/GAN) — Demo + Code

**What this is:** A small generative image project exploring Diffusion and GAN baselines on CelebA (64×64), with a minimal web demo.
**Who it's for:** Recruiters/engineers who want to quickly see working generative AI, and learners who want a clean, reproducible baseline.

**Live Demo:** (paste your Hugging Face Space link here)  
**Project/Portfolio:** (optional link)

## TL;DR
- Diffusion + GAN baselines on CelebA.
- Biggest wins: sane initialization, BN/activation placement, consistent FID evaluation.
- Results snapshot (replace with your numbers):

| Model | Best FID | Epoch | Resolution | Notes |
|------:|---------:|------:|-----------:|-------|
| DDPM  |    —     |   —   |   64×64    | fill later |
| DCGAN |    —     |   —   |   64×64    | optional |
| Glow  |    —     |   —   |   64×64    | optional |

> Hardware/time: e.g., 1×T4 GPU, batch 64, ~X hours per Y epochs.

---

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app/app.py
```

## Project structure
```
gen-ai-face-synthesis/
  README.md
  requirements.txt
  app/
    app.py               # Gradio demo (placeholder now; swap in real sampler next)
  src/
    inference.py         # put your real diffusion/GAN inference here
    utils.py
  models/
    download_weights.py  # (optional) fetch trained weights from HF
  assets/                # screenshots or small sample images
  tests/
    test_smoke.py        # sanity test (imports)
  notebooks/
    ...                  # your training notebooks (read-only)
  LICENSE
  .gitignore
```

## How to wire your real model
- Move your **sampling** function from a notebook/script into `src/inference.py` as `run_inference(seed: int) -> PIL.Image`.
- If you have trained weights, do **not** commit them. Publish to HF Model Hub or add a tiny downloader in `models/download_weights.py`.
- In `app/app.py`, import `run_inference` and return that image instead of the placeholder.

## Data & license
- Trained on **CelebA** (link + license).  
- Code is MIT-licensed.