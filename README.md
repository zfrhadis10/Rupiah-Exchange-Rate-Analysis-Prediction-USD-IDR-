# 📊 Rupiah Exchange Rate Analysis & Prediction (USD/IDR)
### Daily Multi-Asset Macroeconomic & Financial Time Series — Indonesia & Global Markets

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project analyzes and predicts the Indonesian Rupiah exchange rate against the US Dollar (USD/IDR) using daily historical data from **January 2010 to May 2026** (4,277 rows). The analysis covers the relationship between USD/IDR and 8 global and domestic economic variables, alongside a performance comparison of 4 machine learning models.

**Key questions addressed:**
- Which economic variables correlate most strongly with Rupiah depreciation/appreciation?
- How accurately can machine learning models predict daily exchange rates?
- Why does a simpler model (Linear Regression) outperform complex ones (Random Forest, Gradient Boosting) on this time series data?

---

## 📁 Repository Structure

```
📦 Analisis-Prediksi-Nilai-Tukar-Rupiah-USD-IDR/
├── 📓 analisis_rupiah.ipynb              # Main notebook (EDA + prediction)
├── 📄 README.md                          # Project documentation
├── 📊 data/
│   └── Daily_Multi-Asset_Macroeconomic_and_Financial_Time_Series_Dataset.csv
└── 📈 output/
    ├── 01_tren_historis.png
    ├── 02_korelasi_heatmap.png
    ├── 03_korelasi_usdidr.png
    ├── 04_scatter_usdidr.png
    ├── 05_usdidr_tahunan.png
    ├── 06_volatilitas_usdidr.png
    ├── 07_prediksi_semua_model.png
    ├── 08_prediksi_terbaik.png
    └── 09_feature_importance.png
```

---

## 📦 Dataset

| Attribute | Detail |
|---|---|
| **Source** | Daily Multi-Asset Macroeconomic & Financial Time Series |
| **Period** | January 1, 2010 – May 29, 2026 |
| **Rows** | 4,277 (daily frequency) |
| **Columns** | 10 |
| **Missing Values** | 1 row (handled with forward-fill) |

### Variables

| Column | Description | Unit |
|---|---|---|
| `Date` | Trading date | YYYY-MM-DD |
| `USDIDR` | Rupiah exchange rate vs US Dollar *(prediction target)* | IDR per 1 USD |
| `IHSG` | Jakarta Composite Index — Indonesian stock market | Index points |
| `SP500` | S&P 500 — US stock market index | Index points |
| `GOLD` | World gold price | USD per troy oz |
| `OIL` | Brent crude oil price | USD per barrel |
| `VIX` | CBOE Volatility Index — global market uncertainty gauge | Index |
| `CPI` | Consumer Price Index — Indonesia inflation | % |
| `BI_rate` | Bank Indonesia benchmark interest rate | % |
| `US_rate` | US Federal Reserve benchmark interest rate | % |

---

## 🔧 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/zfrhadis10/Analisis-Prediksi-Nilai-Tukar-Rupiah-USD-IDR-.git
cd Analisis-Prediksi-Nilai-Tukar-Rupiah-USD-IDR-
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Run the Notebook
```bash
jupyter notebook analisis_rupiah.ipynb
```

> **Note:** Make sure the dataset file is placed in the same directory as the notebook, or adjust the file path in the data loading cell.

---

## 🗂️ Analysis Workflow

The notebook is divided into 3 main sections:

```
Section 1: Data Exploration & Cleaning
    ├── Check missing values → forward-fill
    ├── Descriptive statistics for all variables
    └── Historical trend visualization for all 9 variables (2010–2026)

Section 2: Multi-Asset Analysis
    ├── Correlation heatmap across all variables
    ├── Variable-specific correlation with USDIDR
    ├── Scatter plots — 4 strongest variables vs USDIDR
    ├── Annual USDIDR movement + major event annotations
    └── 30-day rolling volatility analysis

Section 3: USDIDR Prediction
    ├── Feature engineering (lag features, rolling mean, rate differential)
    ├── Train-test split (80:20, time-ordered — no shuffling)
    ├── Training & evaluation of 4 ML models
    ├── Prediction vs actual visualization
    └── Feature importance analysis (Random Forest)
```

---

## ⚙️ Feature Engineering

Additional derived features were created to improve prediction performance:

| Feature | Construction | Purpose |
|---|---|---|
| `USDIDR_lag1`, `lag7`, `lag30` | USDIDR value at t-1, t-7, t-30 days | Capture short-term exchange rate inertia |
| `IHSG_lag1`, `SP500_lag1` | Index value at t-1 | Previous day's market signal |
| `USDIDR_ma7`, `ma30` | 7 & 30-day rolling mean | Capture short & medium-term trends |
| `rate_diff` | `BI_rate` − `US_rate` | Indonesia–US interest rate differential |
| `Month`, `DayOfWeek` | Month (1–12) & day of week (0–6) | Capture seasonal patterns |

**Data split:** 80% for training (2010–mid 2023), 20% for testing (mid 2023–May 2026). Temporal order is preserved — no random shuffling.

---

## 📊 Model Results & Performance

### Comparison of 4 Machine Learning Models

| Model | MAE (IDR) | RMSE | R² | MAPE (%) |
|---|---|---|---|---|
| 🥇 **Linear Regression** | **129** | ~210 | **0.9419** | **0.792** |
| 🥈 Ridge Regression | 214 | ~350 | 0.8481 | ~1.35 |
| ❌ Random Forest | 509 | ~640 | -0.0186 | ~3.20 |
| ❌ Gradient Boosting | 653 | ~780 | -0.5817 | ~4.10 |

> **MAE** = average absolute difference between predicted and actual values (IDR) — lower is better  
> **R²** = proportion of variance explained by the model (0–1, closer to 1 is better; negative = worse than a naive mean prediction)  
> **MAPE** = mean absolute percentage error

### Why Does Linear Regression Win?

This result seems counterintuitive since complex models usually outperform simple ones. Here's the explanation:

- **USD/IDR follows a strong long-term linear trend** — consistently rising from ~IDR 9,000 (2010) to IDR 17,000+ (2026), a pattern that Linear Regression handles exceptionally well
- **Extrapolation problem in tree-based models** — Random Forest and Gradient Boosting cannot predict values outside the training range. When the exchange rate exceeded IDR 16,000 (never seen during training), tree models stalled and produced overly conservative predictions (~IDR 15,500)
- **Takeaway:** For time series with a strong, monotonic trend, simpler models are often more robust than complex ensemble methods

---

## 🔍 Key Findings

### 1. Correlation with USDIDR
| Variable | Correlation | Brief Interpretation |
|---|---|---|
| IHSG | +0.87 | Both grew together over the long term alongside Indonesia's economic growth |
| SP500 | +0.85 | US market growth co-moves with long-term Rupiah depreciation |
| US_rate | +0.68 | Fed rate hikes → capital outflow from Indonesia → Rupiah weakens |
| GOLD | +0.53 | Positive long-term co-trend, but not direct causality |
| CPI | -0.67 | Higher inflation occurred during the strong Rupiah era (2010–2014) |
| OIL | -0.43 | High oil prices coincide with global economic booms → Rupiah relatively stable |
| BI_rate | -0.41 | BI Rate was higher during the strong Rupiah era — era effect, not direct causality |
| VIX | -0.00 | No long-term correlation — VIX impact on Rupiah is short-lived |

### 2. Feature Importance (Random Forest)

| Feature | Importance Score | Meaning |
|---|---|---|
| `USDIDR_ma7` | **0.655** | 65.5% of predictions driven by the 7-day rolling average |
| `USDIDR_lag1` | 0.320 | 32% driven by yesterday's exchange rate |
| `USDIDR_ma30` | 0.014 | 1.4% from the 30-day trend |
| All others | < 0.001 | OIL, GOLD, IHSG, etc. contribute almost nothing |

> This supports the **Random Walk Hypothesis** — in the short term, exchange rates are most determined by their own recent history, not external macroeconomic factors.

### 3. Highest Rupiah Volatility Periods
- **2020** — COVID-19 pandemic (Rupiah briefly hit IDR 16,800)
- **2014–2015** — Taper Tantrum & emerging market crisis
- **2011–2012** — Data anomalies / outliers detected in the dataset

---

## 📈 Output Visualizations

| # | Chart | Description |
|---|---|---|
| 01 | Historical Trends | 9 panels — all variables from 2010–2026 |
| 02 | Correlation Heatmap | Full correlation matrix across all variables |
| 03 | Correlation vs USDIDR | Bar chart — strength of each variable's relationship with USDIDR |
| 04 | Scatter Plots | 4 strongest variables vs USDIDR with trend lines |
| 05 | Annual USDIDR | Yearly average + min-max range + major event annotations |
| 06 | Volatility | Exchange rate + 30-day rolling std (2010–2026) |
| 07 | All Model Predictions | All 4 models vs actual values |
| 08 | Best Model | Linear Regression predictions + residual panel |
| 09 | Feature Importance | Each feature's contribution in Random Forest |

---

## ⚠️ Limitations

- **Not for trading** — this project is for analytical and educational purposes only, not a trading signal
- **Excluded factors:** government fiscal policy, Bank Indonesia FX market intervention, geopolitical sentiment, and black swan events
- **Data anomalies:** Extreme spikes detected in 2011–2012 USDIDR data that affect some analyses
- **Extrapolation:** Prediction models are only valid within the range of values seen during training. For future forecasting, additional approaches are needed (LSTM, ARIMA-X, etc.)

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥1.5 | Data manipulation and analysis |
| `numpy` | ≥1.23 | Numerical computation |
| `matplotlib` | ≥3.6 | Chart and visualization |
| `seaborn` | ≥0.12 | Heatmap and plot styling |
| `scikit-learn` | ≥1.1 | ML models, preprocessing, evaluation |
| `jupyter` | ≥1.0 | Notebook environment |

---

## 👤 Author

**Zafirah Aida Adista**  
Data Analyst · Universitas Airlangga · Hacktiv8 Comprehensive Data Analytics

[![GitHub](https://img.shields.io/badge/GitHub-zfrhadis10-181717?logo=github)](https://github.com/zfrhadis10)

---

## 📄 License

This project was built for portfolio and educational purposes. The dataset is publicly available.
