# Water Quality Index Assessment of Haryana, India
### A Machine Learning Framework for Groundwater and Surface Water Potability

> **Research Project** — Department of Computer Science & Engineering,Guru Nanak Dev University ,Regional Campus Gurdaspur   
> **Supervisor:** Dr. Harjot Kaur, Assistant Professor  
> **Author:** Sukhmanpreet Kaur

---

## Overview

This repository contains the complete codebase, data, and visualizations for a research study on **Water Quality Index (WQI) assessment** of groundwater and surface water sources across **14 districts of Haryana, India** (2022–2024). The study integrates traditional WQI computation with modern machine learning pipelines to classify water potability, predict WQI values, and identify spatial risk zones — all benchmarked against **BIS IS 10500:2012** drinking water standards.

The dataset was curated and provided by the supervising professor as part of the research collaboration.

---

## Research Highlights

- **WQI computed** for 42 groundwater records (14 sites × 3 years) and 36 surface water records (12 sites × 3 years) using the weighted arithmetic mean method
- **6 ML classifiers** evaluated: KNN, Random Forest, SVM, Gradient Boosting, Decision Tree, XGBoost + K-Means clustering
- **5 ML regressors** evaluated: Random Forest, Gradient Boosting, Decision Tree, XGBoost, SVR
- **Data augmentation pipeline**: Gaussian noise injection → KNN imputation → StandardScaler → SMOTE class balancing
- **22 interactive Plotly visualizations** (HTML) + **13 publication-quality static figures** (PNG)
- Community health impact assessed via field survey (50 respondents, anonymized)

---

## Repository Structure

```
├── 📓 Notebooks (see table below)
│
├── 📊 Data
│   ├── Data Sheet Ground Water.csv       ← Raw groundwater measurements
│   ├── Data Sheet Surface Water.csv      ← Raw surface water measurements
│   ├── groundwater_wqi.csv               ← Computed GW WQI (all sites/years)
│   ├── surface_wqi.csv                   ← Computed SW WQI
│   ├── gclassification_metrics.csv       ← GW classifier performance
│   ├── gregression_metrics.csv           ← GW regressor performance
│   ├── sclassification_metrics_summary.csv ← SW classifier performance
│   ├── sregression_metrics_summary.csv   ← SW regressor performance
│   ├── gw_risk_index.csv                 ← Spatial risk scores (GW)
│   ├── surface_relative_risk.csv         ← Spatial risk scores (SW)
│   └── WQI Final Results Table.csv       ← Consolidated ML results
│
├── 📁 research paper/
│   ├── generate_plots.py                 ← Generates 13 static PNG figures
│   ├── interactive_eda.py                ← Generates 22 interactive HTML plots
│   ├── plots/                            ← Static PNG figures (Fig1–Fig13)
│   └── interactive_plots/               ← Interactive HTML visualizations
│
└── README.md
```

---

## Notebooks

| # | Notebook | Purpose | Keep Output? |
|---|----------|---------|:---:|
| 1 | `groundwater_wqi.ipynb` | WQI computation for groundwater — weighted arithmetic mean, BIS parameter weighting, class assignment | ✅ Yes |
| 2 | `surfacewater_wqi.ipynb` | WQI computation for surface water | ✅ Yes |
| 3 | `groundclassification.ipynb` | ML classification on GW — trains KNN, RF, SVM, GB, DT, XGBoost; outputs metrics, confusion matrices | ✅ Yes |
| 4 | `groundregression.ipynb` | ML regression on GW — predicts continuous WQI; outputs R², RMSE, MAE, NSE | ✅ Yes |
| 5 | `surfaceclassification.ipynb` | ML classification on SW | ✅ Yes |
| 6 | `surfaceregression.ipynb` | ML regression on SW | ✅ Yes |
| 7 | `haryana_combined_wqi_map.ipynb` | Choropleth map of WQI across Haryana districts (Folium/Plotly) | ✅ Yes |
| 8 | `health_alert_system.ipynb` | Rule-based health alert generation from WQI thresholds | ✅ Yes |
| 9 | `augment_gwclassification.ipynb` | Data augmentation + SMOTE pipeline (GW classification) — intermediate step | ⬜ Clear |
| 10 | `augment_gwregression.ipynb` | Data augmentation pipeline (GW regression) — intermediate step | ⬜ Clear |
| 11 | `augment_swclassification.ipynb` | Data augmentation + SMOTE pipeline (SW classification) | ⬜ Clear |
| 12 | `augment_swregression.ipynb` | Data augmentation pipeline (SW regression) | ⬜ Clear |
| 13 | `generate_survey.ipynb` | Survey data generation notebook | ⬜ Clear |

> **✅ Keep Output** = render the notebook with all cell outputs visible on GitHub  
> **⬜ Clear** = run *Kernel → Restart & Clear Output* before committing — these are utility/preprocessing notebooks with large intermediate arrays that add noise

---

## Methodology

```
Raw Data (BIS 8 Parameters)
        │
        ▼
  WQI Computation
  (Weighted Arithmetic Mean, IS 10500:2012)
        │
        ▼
  Data Augmentation
  Gaussian Noise (1% std, ×3) → KNN Imputation (k=5) → StandardScaler
        │
        ▼
  Class Balancing (SMOTE)
        │
        ├──── Classification ────► KNN · RF · SVM · GB · DT · XGBoost · K-Means
        │                          Metrics: Accuracy · Precision · Recall · F1 · MCC
        │
        └──── Regression ─────► RF · GB · DT · XGBoost · SVR
                                 Metrics: R² · RMSE · MAE · NSE
```

**WQI Classes (BIS IS 10500:2012):**

| WQI Range | Groundwater Class | Surface Water Class |
|-----------|------------------|-------------------|
| ≤ 50 | Excellent | — |
| 51–100 | Good | Medium |
| 101–200 | Poor | Poor |
| 201–300 | Very Poor | Very Poor |
| > 300 | Unsuitable | Unsuitable |

---

## Key Parameters Monitored

pH · Turbidity · Electrical Conductivity · Chloride · Sulphates · Iron · COD · BOD · DO · Ammonia · Nitrate · Total Bacterial Count · Total Fungal Count

---

## How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn plotly python-docx pdfplumber folium scipy

# 3. Run WQI notebooks first (they produce the *_wqi.csv files)
jupyter notebook groundwater_wqi.ipynb
jupyter notebook surfacewater_wqi.ipynb

# 4. Run classification and regression notebooks
jupyter notebook groundclassification.ipynb
# ... (repeat for other notebooks)

# 5. Generate all figures
cd "research paper"
python generate_plots.py          # → plots/ (13 static PNGs)
python interactive_eda.py         # → interactive_plots/ (22 HTML files)

---

## Visualizations

### Static Figures (`research paper/plots/`)
| Figure | Description |
|--------|-------------|
| Fig1 | ML Framework Overview |
| Fig2 | Parameter Histograms with KDE curves |
| Fig3 | Pearson Correlation Heatmaps (GW & SW) |
| Fig4 | Box Plots with BIS IS 10500:2012 thresholds |
| Fig5 | K-Means Clustering Scatter (k=2) |
| Fig6–10 | Classifier Comparisons: Accuracy · Precision · Recall · F1 · MCC |
| Fig11 | Model Complexity Radar Chart |
| Fig12a/b | Confusion Matrices — Groundwater & Surface Water |
| Fig13 | Combined Metrics Dashboard |

### Interactive Plots (`research paper/interactive_plots/`)
Open any `.html` file directly in a browser — no server required.

`EDA1` WQI distributions · `EDA2` Parameter histograms · `EDA3a/b` Correlation heatmaps · `EDA4` BIS box plots · `EDA5` Temporal WQI trends · `EDA6a/b` K-Means PCA scatter · `CLS1a/b` Confusion matrices · `CLS2` Metric bars · `CLS3` Radar chart · `REG1–4` Regression diagnostics & feature importance

---

## Dependencies

| Library | Version (tested) | Purpose |
|---------|-----------------|---------|
| pandas | ≥ 1.5 | Data manipulation |
| numpy | ≥ 1.23 | Numerical computation |
| scikit-learn | ≥ 1.2 | ML models, preprocessing |
| imbalanced-learn | ≥ 0.10 | SMOTE oversampling |
| xgboost | ≥ 1.7 | XGBoost classifier/regressor |
| matplotlib / seaborn | ≥ 3.6 / 0.12 | Static visualizations |
| plotly | ≥ 5.13 | Interactive HTML plots |
| python-docx | ≥ 0.8 | Research paper generation |
| scipy | ≥ 1.10 | Statistical functions |

---

## Dataset

The dataset was **provided by the supervising professor** and covers water quality monitoring data from **14 districts across Haryana, India** over three sampling years (2022, 2023, 2024). It includes physicochemical and microbiological parameters for both groundwater (bore wells, hand pumps) and surface water (rivers, canals, ponds) sources.

> ⚠️ **Data Usage:** This dataset was shared exclusively for academic research purposes under university guidelines. Redistribution or commercial use is not permitted without prior consent from the data provider.

---

## Privacy

The community health impact survey involved **50 respondents** across 14 sampling sites. All respondent identities have been anonymized (HS001–HS050) in the published materials in compliance with academic ethical standards.

---

## Citation

If you reference this work, please cite as:

```
Kaur, S. (2024). Water Quality Index Assessment of Haryana, India:
A Machine Learning Framework for Groundwater and Surface Water Potability.
[University Name], Department of [Department]. Supervised by [Professor Name].
```

---

## Acknowledgements

- **Supervisor:** Dr. Harjot Kaur — for providing the dataset, guidance, and domain expertise throughout this research
- **Guru Nanak Dev University, Regional Campus, Gurdaspur** — for providing the academic infrastructure and resources
- BIS IS 10500:2012 — Bureau of Indian Standards drinking water guidelines
- UN SDG 6 — Clean Water and Sanitation, for contextualizing the global relevance of this work

---

*This project was conducted as part of an academic research program. The code and outputs are shared for transparency and reproducibility.*
