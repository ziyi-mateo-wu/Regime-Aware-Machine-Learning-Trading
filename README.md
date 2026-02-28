# Regime-Aware Machine Learning System for Market Strategy Prediction 📈

> A quantitative research framework examining how volatility regimes shape the predictive performance of machine learning models in financial markets.

[![Status](https://img.shields.io/badge/Strategy-Volatility%20Filtering-blueviolet?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Performance](https://img.shields.io/badge/Calm%20Regime%20Acc-54.93%25-green?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Language](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)

---

## 1. Executive Summary

Financial markets are **heteroscedastic**—long periods of stability are interrupted by volatility shocks. Traditional ML models often assume stationarity and therefore fail during turbulent regimes.

This project develops a **Regime-Aware Machine Learning System** that:
- Segments markets into **volatility regimes** using a statistically grounded threshold.
- Predicts next-day SPY direction using engineered trend, momentum, and volatility features.
- Converts a marginal 51.30% classifier into a **regime-conditional trading signal**.
- Reveals a **+4.23% accuracy spread** between calm and volatile regimes.

> **Core Insight:** The model’s predictive edge exists *only* in low-volatility environments. In high volatility, technical signals lose meaning.

---

## 2. Context & Motivation 💡

Inspired by practices in **Credit Risk Control (HSBC)**, where models must remain robust under structural breaks, this project applies similar principles to market prediction.

Instead of treating prediction as a static task, the system implements a **“Fair Weather Alpha”** approach:
- **Trade only when the regime supports the model’s assumptions**.
- **Stay in cash when volatility invalidates technical structure**.

---

## 3. Project Architecture 🏗️

The pipeline is engineered in four phases:

### Phase 1 — Quantitative Feature Engineering
- **Trend:** SMA(50), SMA(200)
- **Momentum:** Custom RSI (Wilder’s Smoothing via EWM)
- **Volatility:** 20-day rolling standard deviation

### Phase 2 — System Integrity & Blind Calibration
- **Anti-leakage:** Volatility threshold calibrated *only* on training data.
- **Local Persistence:** CSV caching for reproducibility.
- **Causality:** Strict chronological alignment and shift(-1) target engineering.

### Phase 3 — Adaptive Ensemble Modeling
- **Logistic Regression:** Linear baseline fitted to log-returns.
- **Random Forest:** Non-linear challenger with `max_depth=5` to prevent overfitting.

---

## 4. Mathematical Framework 📐

### 4.1 Dynamic Regime Classification (Blind Calibration)

To ensure methodological rigor, the regime threshold $\theta$ is derived exclusively from the training-set median (2014–2021):

$$
R_t = 
\begin{cases} 
\text{Volatile}, & \sigma_t > \theta_{train} \\
\text{Stable}, & \sigma_t \leq \theta_{train}
\end{cases}
$$

Where:
- $\sigma_t$ = 20-day rolling volatility.
- $\theta_{train} = 0.00701$.

### 4.2 Target Engineering (Preventing Look-Ahead Bias)

The target $Y_t$ is aligned with the next day's return $r_{t+1}$ using a strict chronological shift:

$$
Y_t = \mathbb{I}(r_{t+1} > 0)
$$

---

## 5. Performance & Alpha Discovery 📊

### 5.1 Model Comparison (Test Set: 2022–2024)

| Metric | Logistic Regression | Random Forest |
| :--- | :--- | :--- |
| **Accuracy** | 49.50% | **51.30%** |
| **Behavior** | Structural "Always-Buy" Bias | Captures non-linear signals |

### 5.2 Regime Spread (Key Finding)

| Regime | Condition | Accuracy | Observations ($N$) |
| :--- | :--- | :--- | :--- |
| **Low Volatility** | $\sigma_t \leq 0.00701$ | **54.93%** | 71 |
| **High Volatility** | $\sigma_t > 0.00701$ | 50.70% | 430 |

---

## 6. Visual Analysis 📉

### 6.1 SPY 10-Year Structural Context (2014-2024)
<img width="100%" src="https://github.com/user-attachments/assets/3805d9b4-6b7d-42e5-9cb7-abaf01de1201" />

### 6.2 Regime Component: Rolling Volatility (Market Fear)
<img width="100%" src="https://github.com/user-attachments/assets/c152e344-d49c-4925-9000-d2aa0a46bc0e" />

### 6.3 Strategy Resilience Check: Accuracy by Regime
<img width="100%" src="https://github.com/user-attachments/assets/a85f0d3b-c562-40c9-b50a-9cc7decb01b1" />

---

## 7. Developer Reflection: Bridging Theory & Industry

This project transitions from "scripting" to **financial engineering** by enforcing:
- **Reproducibility:** Enforcing a global random seed (`np.random.seed(42)`) to ensure deterministic performance.
- **Vectorization:** Leveraging Pandas `ewm` and vectorized arithmetic for institutional-grade indicator speed.
- **Methodological Rigor:** Implementing "Blind Calibration" to prevent Look-Ahead Bias and ensure out-of-sample integrity.

---

## 8. Repository Structure
- `Regime_Aware_ML_Strategy.ipynb`: Full implementation, data audit, and reflection.
- `spy_data_fixed_10y.csv`: Local data cache (2014-2024).
- `README.md`: Executive research summary.
