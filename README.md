# Regime-Aware Machine Learning System for Market Strategy Prediction 📈

> A quantitative research framework examining how volatility regimes shape the predictive performance of machine learning models in financial markets.

[![Status](https://img.shields.io/badge/Strategy-Volatility%20Filtering-blueviolet?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Performance](https://img.shields.io/badge/Calm%20Regime%20Acc-54.93%25-green?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Language](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)

---

## 1. Executive Summary

[cite_start]Financial markets are **heteroscedastic**—long periods of stability are interrupted by volatility shocks[cite: 198, 200]. [cite_start]Traditional ML models assume stationarity and therefore fail during turbulent regimes[cite: 199].

This project develops a **Regime-Aware Machine Learning System** that:

- [cite_start]Segments markets into **volatility regimes** using a statistically grounded threshold [cite: 211, 891]
- [cite_start]Predicts next-day SPY direction using engineered trend, momentum, and volatility features [cite: 577, 606]
- [cite_start]Converts a marginal 51% classifier into a **regime-conditional trading signal** [cite: 883, 1060]
- [cite_start]Reveals a **+4.23% accuracy spread** between calm and volatile regimes [cite: 974, 1046]

> [cite_start]**Core Insight:** The model’s predictive edge exists *only* in low-volatility environments[cite: 1044, 1054]. [cite_start]In high volatility, technical signals lose meaning[cite: 1052].

---

## 2. Context & Motivation 💡

[cite_start]Inspired by practices in **Credit Risk Control (HSBC)**, where models must remain robust under structural breaks, this project applies similar principles to market prediction[cite: 188, 401].

[cite_start]Instead of treating prediction as a static task, the system implements a **“Fair Weather Alpha”** approach[cite: 1053]:

- [cite_start]**Trade only when the regime supports the model’s assumptions** [cite: 1059]
- [cite_start]**Stay in cash when volatility invalidates technical structure** [cite: 1057]

---

## 3. Project Architecture 🏗️

The pipeline is engineered in four phases:

### Phase 1 — Quantitative Feature Engineering
- [cite_start]**Trend:** SMA(50), SMA(200) [cite: 436, 437]
- [cite_start]**Momentum:** Custom RSI (Wilder’s Smoothing) [cite: 396, 439]
- [cite_start]**Volatility:** 20-day rolling standard deviation [cite: 442]

### Phase 2 — System Integrity & Blind Calibration
- [cite_start]Anti-leakage volatility threshold [cite: 682, 919]
- [cite_start]Local CSV caching for reproducibility [cite: 255, 261]
- [cite_start]Strict chronological alignment [cite: 584, 608]

### Phase 3 — Adaptive Ensemble Modeling
- [cite_start]Logistic Regression (baseline) [cite: 684, 721]
- [cite_start]Random Forest (non-linear challenger) [cite: 687, 724]

### Phase 4 — Regime-Based Performance Attribution
- [cite_start]Accuracy decomposition by volatility regime [cite: 885, 942]
- [cite_start]Identification of “Fair Weather Alpha” [cite: 1053, 1060]

---

## 4. Mathematical Framework 📐

### 4.1 Dynamic Regime Classification

[cite_start]Market regimes $R_t$ are classified based on rolling volatility $\sigma_t$ relative to the training-set median threshold $\theta_{train}$[cite: 898, 921]:

$$
R_t = 
\begin{cases} 
\text{Volatile}, & \sigma_t > \theta_{train} \\
\text{Stable}, & \sigma_t \leq \theta_{train}
\end{cases}
$$

Where:
- [cite_start]$\sigma_t$ = 20-day rolling volatility [cite: 442, 562]
- [cite_start]$\theta_{train} \approx 0.007$ = median training-set volatility [cite: 923, 965]

### 4.2 Target Engineering

[cite_start]To prevent look-ahead bias, the target $Y_t$ is aligned with the next day's return $r_{t+1}$[cite: 578, 655]:

$$
Y_t = \mathbb{I}(r_{t+1} > 0)
$$

---

## 5. Performance & Alpha Discovery 📊

### [cite_start]5.1 Model Comparison (2022–2024) [cite: 813]

| Metric | Logistic Regression | Random Forest |
| :--- | :--- | :--- |
| **Accuracy** | 49.50% | **51.30%** |
| **Behavior** | [cite_start]"Always Buy" Heuristic [cite: 875] | [cite_start]Captures non-linear signals [cite: 881] |

### [cite_start]5.2 Regime Spread (Key Finding) [cite: 966]

| Regime | Condition | Accuracy | Observations ($N$) |
| :--- | :--- | :--- | :--- |
| **Low Volatility** | $\sigma_t \leq 0.007$ | **54.93%** | [cite_start]71 [cite: 970] |
| **High Volatility** | $\sigma_t > 0.007$ | 50.70% | [cite_start]430 [cite: 972] |

> [cite_start]**Key Finding:** The model demonstrates a tangible statistical edge only in calm markets (+4.23% spread)[cite: 974, 1046].

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

## [cite_start]7. Strategic Implications [cite: 1040]

This system demonstrates that:
- [cite_start]**When to trade** is as important as **what to trade**[cite: 1061].
- [cite_start]Volatility should be used as a **model activation filter**[cite: 1054].
- [cite_start]A weak classifier can become a reliable signal when deployed selectively[cite: 1060].

---

## 8. Installation & Usage 💻

```bash
# 1. Clone the repository
git clone [https://github.com/ziyi-mateo-wu/Regime-Aware-Machine-Learning-Trading.git](https://github.com/ziyi-mateo-wu/Regime-Aware-Machine-Learning-Trading.git)
cd Regime-Aware-Machine-Learning-Trading

# 2. Install dependencies
pip install pandas numpy scikit-learn yfinance matplotlib seaborn
