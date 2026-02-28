# Regime-Aware Machine Learning System for Market Strategy Prediction 📈

> A quantitative research framework examining how volatility regimes shape the predictive performance of machine learning models in financial markets.

[![Status](https://img.shields.io/badge/Strategy-Volatility%20Filtering-blueviolet?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Performance](https://img.shields.io/badge/Calm%20Regime%20Acc-54.93%25-green?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Language](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)

---

## 1. Executive Summary

[cite_start]Financial markets are **heteroscedastic**—long periods of stability are interrupted by volatility shocks[cite: 197, 564]. [cite_start]Traditional ML models assume stationarity and therefore fail during turbulent regimes[cite: 196].

This project develops a **Regime-Aware Machine Learning System** that:
- [cite_start]Segments markets into **volatility regimes** using a statistically grounded threshold[cite: 208, 935].
- [cite_start]Predicts next-day SPY direction using engineered trend, momentum, and volatility features[cite: 207, 383].
- [cite_start]Converts a marginal 51.30% classifier into a **regime-conditional trading signal**[cite: 883, 1076].
- [cite_start]Reveals a **+4.23% accuracy spread** between calm and volatile regimes[cite: 980, 1050].

> [cite_start]**Core Insight:** The model’s predictive edge exists *only* in low-volatility environments[cite: 1048, 1054]. [cite_start]In high volatility, technical signals lose meaning[cite: 1056].

---

## 2. Context & Motivation 💡

[cite_start]Inspired by practices in **Credit Risk Control (HSBC)**, where models must remain robust under structural breaks, this project applies similar principles to market prediction[cite: 1080, 1083].

Instead of treating prediction as a static task, the system implements a **“Fair Weather Alpha”** approach:
- [cite_start]**Trade only when the regime supports the model’s assumptions**[cite: 1058].
- [cite_start]**Stay in cash when volatility invalidates technical structure**[cite: 1064, 1076].

---

## 3. Project Architecture 🏗️

The pipeline is engineered in four phases:

### Phase 1 — Quantitative Feature Engineering
- [cite_start]**Trend:** SMA(50), SMA(200) [cite: 433, 434]
- [cite_start]**Momentum:** Custom RSI (Wilder’s Smoothing via EWM) [cite: 412, 436]
- [cite_start]**Volatility:** 20-day rolling standard deviation [cite: 439, 561]

### Phase 2 — System Integrity & Blind Calibration
- [cite_start]**Anti-leakage:** Volatility threshold calibrated *only* on training data[cite: 924, 1046].
- [cite_start]**Local Persistence:** CSV caching for reproducibility[cite: 253, 258].
- [cite_start]**Causality:** Strict chronological alignment and shift(-1) target engineering[cite: 585, 663].

### Phase 3 — Adaptive Ensemble Modeling
- [cite_start]**Logistic Regression:** Linear baseline fitted to log-returns[cite: 690, 726].
- [cite_start]**Random Forest:** Non-linear challenger with `max_depth=5` to prevent overfitting[cite: 693, 730].

---

## 4. Mathematical Framework 📐

### 4.1 Dynamic Regime Classification (Blind Calibration)

[cite_start]To ensure methodological rigor, the regime threshold $\theta$ is derived exclusively from the training-set median (2014–2021)[cite: 926, 971]:

$$
R_t = 
\begin{cases} 
\text{Volatile}, & \sigma_t > \theta_{train} \\
\text{Stable}, & \sigma_t \leq \theta_{train}
\end{cases}
$$

Where:
- [cite_start]$\sigma_t$ = 20-day rolling volatility[cite: 562].
- [cite_start]$\theta_{train} = 0.00701$[cite: 971].

### 4.2 Target Engineering (Preventing Look-Ahead Bias)

[cite_start]The target $Y_t$ is aligned with the next day's return $r_{t+1}$ using a strict chronological shift[cite: 580, 662]:

$$
Y_t = \mathbb{I}(r_{t+1} > 0)
$$

---

## 5. Performance & Alpha Discovery 📊

### [cite_start]5.1 Model Comparison (Test Set: 2022–2024) [cite: 591, 667]

| Metric | Logistic Regression | Random Forest |
| :--- | :--- | :--- |
| **Accuracy** | [cite_start]49.50% [cite: 855] | [cite_start]**51.30%** [cite: 857] |
| **Behavior** | [cite_start]Structural "Always-Buy" Bias [cite: 880] | [cite_start]Captures non-linear signals [cite: 886] |

### [cite_start]5.2 Regime Spread (Key Finding) [cite: 972]

| Regime | Condition | Accuracy | Observations ($N$) |
| :--- | :--- | :--- | :--- |
| **Low Volatility** | $\sigma_t \leq 0.00701$ | [cite_start]**54.93%** [cite: 976] | [cite_start]71 [cite: 976] |
| **High Volatility** | $\sigma_t > 0.00701$ | [cite_start]50.70% [cite: 978] | [cite_start]430 [cite: 978] |

> [cite_start]**Strategic Verdict:** The model demonstrates a significant statistical edge in calm markets (+4.23% spread)[cite: 980, 1050].

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

[cite_start]This project transitions from "scripting" to **financial engineering** by enforcing[cite: 1085]:
- [cite_start]**Reproducibility:** Enforcing a global random seed (`np.random.seed(42)`) to ensure deterministic performance[cite: 1087, 1088].
- [cite_start]**Vectorization:** Leveraging Pandas `ewm` and vectorized arithmetic for institutional-grade indicator speed[cite: 1089, 1090].
- [cite_start]**Methodological Rigor:** Implementing "Blind Calibration" to prevent Look-Ahead Bias and ensure out-of-sample integrity[cite: 1091].

---

## 8. Repository Structure
- [cite_start]`Regime_Aware_ML_Strategy.ipynb`: Full implementation, data audit, and reflection[cite: 220, 1068].
- [cite_start]`spy_data_fixed_10y.csv`: Local data cache (2014-2024)[cite: 259, 311].
- `README.md`: Executive research summary.
