# CelebA Generative Baselines — DCGAN, cGAN, DDPM, Glow (64×64)

[![Open In Colab (DDPM)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vishvagor/Gen-ai-face-synthesis/blob/main/notebooks/ddpm_celeba.ipynb)
[![Open In Colab (DCGAN)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vishvagor/Gen-ai-face-synthesis/blob/main/notebooks/dcgan_celeba.ipynb)
[![Open In Colab (cGAN)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vishvagor/Gen-ai-face-synthesis/blob/main/notebooks/cgan_celeba.ipynb)
[![Open In Colab (Glow)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vishvagor/Gen-ai-face-synthesis/blob/main/notebooks/glow_celeba.ipynb)

> **No reruns needed.** Each Colab opens with results displayed in the first cell. Training/sampling cells are optional and heavy.

## TL;DR
- Trained **DCGAN, cGAN, DDPM, Glow** on **CelebA 64×64** as practical baselines.
- Biggest wins: sane init, BatchNorm/activation placement, consistent FID evaluation pipeline.
- DDPM produced cleaner faces; DCGAN trained faster; Glow required careful stability tricks.

### Results (fill your numbers)
| Model | Best FID | Epoch | Resolution | Approx Train Time |
|------:|---------:|------:|-----------:|------------------:|
| DCGAN | **443.5**| **50**| 64×64      |  ~10+ hrs (GPU)   |
| cGAN  | **500**  | **50**| 64×64      |  ~10+ hrs (GPU)   |
| DDPM  | **139.6**| **300**| 64×64     | ~10+ hrs (GPU)   |
| Glow  | **269.3**| **50** | 64×64     | ~10+ hrs (GPU)   |

> Metric: **FID** on consistent preprocessing. Hardware notes in each notebook.

## Sample Grids
**DDPM**
<img src="assets/ddpm_grid.png" width="420"/>

**DCGAN**
<img src="assets/dcgan_grid.png" width="420"/>

**cGAN**
<img src="assets/cgan_grid.png" width="420"/>

**Glow**
<img src="assets/glow_grid.png" width="420"/>

<!-- Optional: FID curves if you have them
## FID snapshots
<img src="assets/ddpm_fid.png" width="360"/> <img src="assets/dcgan_fid.png" width="360"/>
-->

## Reproduce (quick)
```bash
git clone https://github.com/Vishvagor/Gen-ai-face-synthesis
cd Gen-ai-face-synthesis
pip install -r requirements.txt
# open notebooks in /notebooks or click the Colab badges above


## Normalizing Flow (Glow-style) code
- Code added under `src/nf/`:
  - `model.py`
  - `training.py`
- You can create a small `infer_nf.py` later if you want a NF demo.
