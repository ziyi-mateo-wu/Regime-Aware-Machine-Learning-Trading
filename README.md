# Regime-Aware Machine Learning System for Market Strategy Prediction 📈

> A quantitative research framework examining how volatility regimes shape the predictive performance of machine learning models in financial markets.

[![Status](https://img.shields.io/badge/Strategy-Volatility%20Filtering-blueviolet?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Performance](https://img.shields.io/badge/Calm%20Regime%20Acc-54.93%25-green?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Language](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)

---

## 1. Executive Summary

Financial markets are **heteroscedastic**—long periods of stability are interrupted by volatility shocks. Traditional ML models assume stationarity and therefore fail during turbulent regimes.

This project develops a **Regime-Aware Machine Learning System** that:

- Segments markets into **volatility regimes** using a statistically grounded threshold  
- Predicts next-day SPY direction using engineered trend, momentum, and volatility features  
- Converts a marginal 51% classifier into a **regime-conditional trading signal**  
- Reveals a **+4.23% accuracy spread** between calm and volatile regimes  

> **Core Insight:** The model’s predictive edge exists *only* in low-volatility environments.  
> In high volatility, technical signals lose meaning.

---

## 2. Context & Motivation 💡

Inspired by practices in **Credit Risk Control (HSBC)**, where models must remain robust under structural breaks, this project applies similar principles to market prediction.

Instead of treating prediction as a static task, the system implements a **“Fair Weather Alpha”** approach:

- **Trade only when the regime supports the model’s assumptions**
- **Stay in cash when volatility invalidates technical structure**

---

## 3. Project Architecture 🏗️

The pipeline is engineered in four phases:

### Phase 1 — Quantitative Feature Engineering
- **Trend:** SMA(50), SMA(200)  
- **Momentum:** Custom RSI (Wilder’s Smoothing)  
- **Volatility:** 20-day rolling standard deviation  

### Phase 2 — System Integrity & Blind Calibration
- Anti-leakage volatility threshold  
- Local CSV caching for reproducibility  
- Strict chronological alignment  

### Phase 3 — Adaptive Ensemble Modeling
- Logistic Regression (baseline)  
- Random Forest (non-linear challenger)  

### Phase 4 — Regime-Based Performance Attribution
- Accuracy decomposition by volatility regime  
- Identification of “Fair Weather Alpha”  

---

## 4. Mathematical Framework 📐

### 4.1 Regime Classification



\[
R_t =
\begin{cases}
\text{Volatile}, & \sigma_t > \theta_{\text{train}} \\
\text{Calm}, & \sigma_t \le \theta_{\text{train}}
\end{cases}
\]



Where:

- \( \sigma_t \) = rolling volatility  
- \( \theta_{\text{train}} \approx 0.007 \) = median training-set volatility  

### 4.2 Target Engineering



\[
Y_t = \mathbb{I}(r_{t+1} > 0)
\]



Aligned strictly to prevent **Look-Ahead Bias**.

---

## 5. Performance & Alpha Discovery 📊

### 5.1 Model Comparison (2022–2024)

| Metric | Logistic Regression | Random Forest |
|-------|---------------------|---------------|
| Accuracy | 49.50% | **51.30%** |
| Behavior | “Always Buy” | Captures non-linear signals |

### 5.2 Regime Spread (Key Finding)

| Regime | Condition | Accuracy | N |
|--------|-----------|----------|---|
| **Low Volatility** | \( \sigma_t \le 0.007 \) | **54.93%** | 71 |
| **High Volatility** | \( \sigma_t > 0.007 \) | 50.70% | 430 |

> **The model only works in calm markets.**  
> High volatility destroys signal quality.

---

## 6. Visual Analysis 📉

### 6.1 SPY 10-Year Structural Context  
<img width="100%" src="https://github.com/user-attachments/assets/3805d9b4-6b7d-42e5-9cb7-abaf01de1201" />

### 6.2 Why These Features Matter  
<img width="100%" src="https://github.com/user-attachments/assets/4f2b87e5-320f-4fe8-a667-9ab00e430ada" />

### 6.3 Regime-Based Performance  
<img width="100%" src="https://github.com/user-attachments/assets/c152e344-d49c-4925-9000-d2aa0a46bc0e" />

### 6.4 Confusion Matrix Comparison  
<img width="100%" src="https://github.com/user-attachments/assets/a85f0d3b-c562-40c9-b50a-9cc7decb01b1" />

---

## 7. Strategic Implications

This system demonstrates that:

- **When to trade** is as important as **what to trade**  
- Volatility should be treated as a **model activation filter**  
- A weak classifier can become useful when deployed selectively  

### Practical Interpretation

- **Risk-Off:** \( \sigma_t > 0.007 \) → Stay in cash  
- **Risk-On:** \( \sigma_t \le 0.007 \) → Use model predictions  

---

## 8. How to Run & View 💻

### Option 1 — Quick View (Recommended)
Open the notebook directly on GitHub:

**`Regime_Aware_ML_Strategy.ipynb`**

All plots and results are pre-rendered.

---

### Option 2 — Local Execution

```bash
# 1. Clone the repository
git clone https://github.com/ziyi-mateo-wu/Regime-Aware-Machine-Learning-Trading.git
cd Regime-Aware-Machine-Learning-Trading

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy scikit-learn yfinance matplotlib seaborn
```

## 9. Repository Structure
Regime-Aware-Machine-Learning-Trading/
│── Regime_Aware_ML_Strategy.ipynb
│── spy_data_fixed_10y.csv
│── README.md
└── (optional) figures/

# 4. Run the notebook
jupyter notebook Regime_Aware_ML_Strategy.ipynb
