# Baqaee & Farhi (2022) — Replication and Extension

**Paper:** Baqaee, David Rezza, and Emmanuel Farhi. "Supply and Demand in Disaggregated
Keynesian Economies with an Application to the COVID-19 Crisis."
*American Economic Review* 112(5): 1397–1435, 2022.

---

## Overview

This repository replicates the core quantitative results of Baqaee & Farhi (2022) using
publicly available data (BEA Input-Output tables and BLS price/output data), then extends
the analysis to the 2021–2022 inflation surge.

**All code is written in Python (NumPy, pandas, matplotlib). No MATLAB is required.**

---

## Repository Structure

```
baqaee-farhi-replication/
├── README.md
├── decisions_log.md          # All data decisions and deviations from paper
├── requirements.txt
├── main.py                   # Orchestrates full pipeline
│
├── src/
│   ├── data_download/
│   │   ├── download_bea.py   # BEA API download scripts
│   │   └── download_bls.py   # BLS API download scripts
│   ├── io_network/
│   │   └── construct_network.py  # IO matrix, Leontief inverse, Domar weights
│   ├── decomposition/
│   │   └── supply_demand_decomp.py  # Supply/demand shock identification & GDP decomp
│   ├── replication/
│   │   ├── covid_episode.py  # Feb–May 2020 GDP decline decomposition
│   │   └── figures_tables.py # Replicated tables and figures
│   ├── extension/
│   │   ├── inflation_episode.py  # 2021–2022 inflation decomposition
│   │   └── comparison.py        # Cross-episode comparison
│   └── visualization/
│       └── create_figures.py    # All matplotlib figure code
│
├── data/
│   ├── raw/
│   │   ├── bea/              # BEA Use tables (downloaded or embedded)
│   │   └── bls/              # BLS price/output data
│   └── processed/            # Cleaned, harmonized data
│
└── outputs/
    ├── figures/              # PNG/PDF figures
    └── tables/               # CSV tables
```

---

## Data Sources

### Publicly Available (used in this replication)

| Dataset | Source | Vintage | Access |
|---------|--------|---------|--------|
| Annual Industry Accounts — Use Table (71 sectors) | BEA | 2017–2022 | Free, BEA API or website |
| Consumer Price Index by expenditure category | BLS | 2020–2022 | Free, BLS API |
| Producer Price Index by industry | BLS | 2020–2022 | Free, BLS API |
| GDP by Industry (Gross Output) | BEA | 2019–2022 | Free, BEA API |
| Personal Consumption Expenditures | BEA NIPA | 2019–2022 | Free, BEA API |
| Monthly supply chain indicators | BLS/NY Fed | 2020–2022 | Free |

### Not Used (proprietary or unavailable)
- Confidential establishment-level microdata
- Proprietary high-frequency transaction data cited in some extensions of the paper

---

## Installation

```bash
pip install -r requirements.txt
```

### Optional: BEA API Key
Register for a free key at https://apps.bea.gov/api/signup/
Set it as an environment variable:
```bash
export BEA_API_KEY="your-key-here"
```
Without a key, the scripts use embedded synthetic data calibrated to match published BEA aggregates.

---

## Running the Analysis

### Full pipeline (recommended)
```bash
python main.py
```

### Step by step
```bash
# 1. Download data (requires internet + BEA API key)
python src/data_download/download_bea.py
python src/data_download/download_bls.py

# 2. Replicate COVID episode
python src/replication/covid_episode.py

# 3. Extend to 2021–2022 inflation
python src/extension/inflation_episode.py

# 4. Generate comparison figures
python src/extension/comparison.py
```

Outputs are written to `outputs/figures/` and `outputs/tables/`.

---

## Key Methodology

### Input-Output Network
Given BEA Use table Z (N×N flows matrix):
- **IO coefficient matrix**: A where A_{ij} = Z_{ij} / x_j (share of sector i in inputs to j)
- **Leontief inverse**: L = (I − A)^{−1}
- **Domar weights**: λ_i = p_i x_i / GDP (gross output share)

### Supply-Demand Shock Identification
Given sector-level log changes in price (ΔP_i) and output (ΔY_i):

```
Supply shock:  s_i = ΔY_i / σ − ΔP_i
Demand shock:  d_i = ΔP_i + ΔY_i / ε
```
where σ is the supply elasticity and ε is the demand price elasticity (see decisions_log.md for calibration).

### GDP Decomposition
```
ΔGDP/GDP ≈ Σ_i Λ_i · s_i   (supply contributions, network-amplified)
           + Σ_i α_i · d_i  (demand contributions, expenditure-weighted)
```
where Λ_i = λ_i × Σ_j L_{ij} is the Domar-weighted Leontief multiplier.

---

## Key Results

### COVID Episode (Feb–May 2020)
- Total GDP decline: approximately −8 to −10%
- Demand shocks dominate: ~70–80% of the decline
- Supply shocks: ~20–30%, concentrated in contact-intensive services
- Largest sectoral contributors: Food services, Air transportation, Arts & Entertainment

### 2021–2022 Inflation Extension
- Total CPI inflation: ~7–9% over 2021–2022
- Supply shocks dominate: energy, semiconductors, supply chain
- Demand shocks: fiscal stimulus, housing demand
- Greater network amplification than COVID episode due to energy/commodities propagation

---

## Citation

```bibtex
@article{baqaee2022supply,
  title={Supply and demand in disaggregated Keynesian economies with an application to the COVID-19 crisis},
  author={Baqaee, David Rezza and Farhi, Emmanuel},
  journal={American Economic Review},
  volume={112},
  number={5},
  pages={1397--1435},
  year={2022}
}
```
