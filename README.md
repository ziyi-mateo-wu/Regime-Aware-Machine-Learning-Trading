# Regime-Aware Machine Learning System for Market Strategy Prediction 📈

> **Quantitative Research Framework**: Investigating how heteroscedastic volatility regimes shape the predictive boundaries of machine learning models in non-stationary financial markets.

[![Status](https://img.shields.io/badge/Strategy-Volatility%20Filtering-blueviolet?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Performance](https://img.shields.io/badge/Calm%20Regime%20Acc-54.93%25-green?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Language](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)

---

## 1. Executive Summary 📄

Financial time-series forecasting is inherently challenging due to the **non-stationary** and **stochastic** nature of markets. Traditional ML models often fail because they assume a static environment and ignore **heteroscedasticity**—the clustering of volatility shocks.

This project develops a **Regime-Aware Machine Learning System** that:
- **Quantifies Market States**: Segments markets into "Calm" and "Volatile" regimes using a statistically grounded threshold.
- **Discovers the Alpha Boundary**: Proves that model accuracy soars to **54.93%** in stable conditions but degrades to near-random (50.70%) during shocks.
- **Engineers a "Fair Weather Alpha" Strategy**: Transforms a weak 51.30% classifier into a robust, regime-conditional system by identifying *when* to trade.

---

## 2. Context & Motivation: From Industry to Theory 💡

The conceptual foundation stems from my experience in **HSBC Credit Risk Control and Management**. I observed that institutional models often fail when their underlying assumptions about volatility are violated. 

Instead of treating prediction as a static task, this system implements a **“Fair Weather Alpha”** approach:
- **Trade only when the regime supports the model’s assumptions**.
- **Stay in cash when volatility invalidates technical structure**.

---

## 3. Methodological Rigor: Ensuring System Integrity 🏗️

### 3.1 "Blind Threshold" Calibration (Anti-Leakage)
To strictly prevent **Look-Ahead Bias**, the regime threshold ($\theta$) is derived exclusively from historical data:
- **Calibration**: The median volatility ($\theta_{train} = 0.00701$) is computed using only the **Training Set (2014-2021)**.
- **Zero-Leakage Implementation**: The test phase (2022-2024) remains "blind" to future distributions, ensuring that results reflect a realistic trading scenario.

### 3.2 Adaptive Ensemble Modeling
- **Linear Baseline (Logistic Regression)**: Failed due to a structural **"Always-Buy" bias** hard-coded during the secular bull market of 2014-2021.
- **Non-Linear Challenger (Random Forest)**: Explicitly constrained with `max_depth=5` to force generalization and prevent the model from memorizing financial noise.

---

## 4. Mathematical Framework 📐

### 4.1 Dynamic Regime Classification
Market regimes ($R_t$) are classified based on the rolling volatility ($\sigma_t$) relative to the training median threshold ($\theta_{train}$):

$$
R_t = 
\begin{cases} 
\text{Volatile}, & \text{if } \sigma_t > \theta_{train} \\
\text{Stable}, & \text{if } \sigma_t \leq \theta_{train}
\end{cases}
$$

### 4.2 Target Engineering (Causality Protocol)
The prediction variable is aligned using a strict **shift(-1)** operation to ensure features ($X_t$) only predict the *future* return ($r_{t+1}$):
$$Y_t = \mathbb{I}(r_{t+1} > 0)$$

---

## 5. Alpha Discovery & Visual Interpretation 📊

### 5.1 SPY 10-Year Historical Price (Structural Breaks)
Visualizing the 2020 COVID-19 crash and the 2022 regime shift. The dataset contains **2,516 clean trading days**, identifying structural breaks where static models fail.
<img width="100%" alt="SPY Price" src="https://github.com/user-attachments/assets/3805d9b4-6b7d-42e5-9cb7-abaf01de1201" />

### 5.2 Why These Features Matter (Trend, Momentum, Regime)
- **Trend Dynamics ($SMA_{200}$)**: The 200-day SMA acted as reliable dynamic support (2014-2019). Its decisive breakdown in 2022 signaled a structural regime change.
- **Momentum ($RSI$)**: Custom Wilder’s Smoothing identified the March 2020 liquidity crisis at RSI ~15 (Seller Exhaustion).
- **Regime ($\sigma$)**: Confirms **Volatility Clustering**, proving that models must be context-dependent.
<img width="100%" alt="Feature Engineering Analysis" src="https://github.com/user-attachments/assets/4f2b87e5-320f-4fe8-a667-9ab00e430ada" />

### 5.3 Regime-Based Performance (The Core Finding)
Accuracy decomposition reveals that predictive signals are most reliable when market noise is lowest.
- **Low Volatility**: **54.93% Accuracy**. Endogenous technical patterns exhibit high structural fidelity.
- **High Volatility**: **50.70% Accuracy**. Exogenous shocks render learned signals irrelevant.
<img width="100%" alt="Regime Performance" src="https://github.com/user-attachments/assets/c152e344-d49c-4925-9000-d2aa0a46bc0e" />

### 5.4 Confusion Matrix Comparison
Contrasting the "Always Buy" bias of Logistic Regression against the nuanced, non-linear predictions of Random Forest.
<img width="100%" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/a85f0d3b-c562-40c9-b50a-9cc7decb01b1" />

---

## 6. Strategic Implication: The Volatility Filter

The project proves that **"When to trade" is as important as "What to trade."** Instead of a static predictor, the model functions best as a **Risk-On / Risk-Off Switch**:
1. **Risk-Off (Cash)**: When Rolling Volatility > 0.00701, stay in cash.
2. **Risk-On (Trade)**: When Rolling Volatility ≤ 0.00701, target the **54.93% accuracy** edge.

---

## 7. How to Run & View
- **Quick View**: Open [Regime_Aware_ML_Strategy.ipynb](Regime_Aware_ML_Strategy.ipynb) directly on GitHub to see pre-rendered results.
- **Local Execution**: Clone and install `pandas`, `numpy`, `scikit-learn`, `yfinance`, `matplotlib`, `seaborn`.
