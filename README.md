# TimeBridge-Enhanced-NonStationnary-MTS
**Non-Stationarity Matters for Long-term Time Series Forecasting — with practical enhancements and clean tooling**

<div align="center">

[Paper (arXiv)](https://arxiv.org/abs/2410.04442) ·
[Original Repo](https://github.com/Hank0626/TimeBridge) ·
[ICML 2025 Poster]

</div>


## What’s in this repo?

This is a **research-grade, enhanced implementation** of **TimeBridge**, focused on **non-stationary multivariate time-series forecasting**.

**Enhancements (this repo):**
-  **Diagnostics**: ADF / KPSS / Engle–Granger cointegration utilities (optional module)
-  **Configurable detrending** in Integrated Attention (EMA / moving-average / HP / STL)
-  **Adaptive patch downsampling** options (learned/Top-K/attn-pool)
-  **Cointegration-aware attention priors** (use EG as a soft prior)
-  **Ablation runners** + seeds + clean result tables
-  **Professional tooling**: .gitignore, dataset layout, Windows & Linux commands

> You can use **vanilla TimeBridge** as in the paper, then progressively enable these improvements.


## Updates

- **2025-05-01** — TimeBridge accepted as **ICML 2025 Poster** (original work).
- **2025-04-18** — Released detailed training logs (see `_logs/`, if present).
- **2025-02-11** — Initial public code release (original).
- **2024-10-08** — First arXiv upload of the paper.

> This fork adds diagnostics & training quality-of-life features on top of the original.


##  Environment Setup

> Windows PowerShell (recommended) or Linux/macOS shell.

```bash
# create & activate venv (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# or on Linux/macOS
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# (Optional) verify CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
