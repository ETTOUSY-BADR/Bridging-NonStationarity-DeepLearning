<h1 align="center"> Bridging Classical Non-Stationarity Tests and Deep Learning</h1>

<h3 align="center">
  <strong>Analysis Framework for Non-Stationary Time-Series + Enhanced TimeBridge Implementation</strong>
</h3>


##  Project Overview

This repository accompanies the research project:

**“Bridging Classical Non-Stationarity Tests and Deep Learning:
A Comparative Analysis of Statistical Diagnostics and TimeBridge Representations.”**

It integrates two worlds:

### **1️ Classical Statistical Tests for Non-Stationarity**
(from the article *Analyse des séries temporelles non-stationnaires* :contentReference[oaicite:0]{index=0})

- Augmented Dickey–Fuller (ADF)
- KPSS test
- Priestley–Subba Rao (PSR)
- Spectral invariance tests
- Engle–Granger Cointegration
- Conditional independence tests for non-stationary time series

### **2️ Deep Learning for Non-Stationary Forecasting**
(based on **TimeBridge** from ICML 2025 as described in the article :contentReference[oaicite:1]{index=1})

- Integrated Attention → removes short-term non-stationarity  
- Cointegrated Attention → preserves long-term dependencies  
- Patch Downsampling  
- Patch Embedding  
- State-of-the-art results on ETT, Weather, Traffic, Electricity, Solar, etc.

---

##  What This Repo Contains

This repo is a **clean Python framework** designed to:

###  Implement TimeBridge (PyTorch)  
###  Add statistical diagnostics for non-stationarity  
###  Evaluate datasets with ADF / KPSS / PSR / EG  
###  Study short-term vs long-term effects of non-stationarity  
###  Produce plots, confusion matrices, and analytical reports  
###  Combine classical econometrics with modern deep learning  

---

##  Differences from the Original TimeBridge Repo

- Added **statistical tests** (ADF, KPSS, PSR, Engle–Granger)
- Added **synthetic dataset generator** for stationary / non-stationary series  
- Added **clean dataset pipeline** (ETT, Traffic, Electricity...)
- Added **evaluation framework**
- Cleaned architecture + Windows compatibility
- Professional `.gitignore` + repo structure
- Clear documentation

Original TimeBridge paper → https://arxiv.org/abs/2410.04442  
Original code → https://github.com/Hank0626/TimeBridge

---

## Environment Setup (Windows, Linux, macOS)

```bash
# create & activate venv (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# OR Linux/macOS
source .venv/bin/activate

# install packages
pip install -r requirements.txt

# optional GPU test
python -c "import torch; print(torch.cuda.is_available())"
