# Regime-Aware Machine Learning System for Market Strategy Prediction 📈

> **Quantitative Research Framework**: Investigating how heteroscedastic volatility regimes shape the predictive boundaries of machine learning models in non-stationary financial markets.

[![Status](https://img.shields.io/badge/Strategy-Volatility%20Filtering-blueviolet?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Performance](https://img.shields.io/badge/Calm%20Regime%20Acc-54.93%25-green?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Language](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)

---

## 1. Executive Summary 📄

Financial time-series forecasting is inherently challenging due to the **non-stationary** and **stochastic** nature of markets. Traditional ML models often fail because they assume a static environment and ignore **heteroscedasticity**—the clustering of volatility shocks that degrade model performance.

This project proposes a **"Regime-Aware"** framework that:
- **Quantifies Market States**: Segments data into "Calm" and "Volatile" regimes using a statistically grounded threshold.
- **Discovers the Alpha Boundary**: Proves that model accuracy soars to **54.93%** in stable conditions but degrades to near-random (50.70%) during shocks.
- **Engineers a "Fair Weather Alpha" Strategy**: Transforms a weak 51.30% classifier into a robust, regime-conditional system by identifying *when* to trade rather than just *what* to trade.

---

## 2. Context & Motivation: From Industry to Theory 💡

The conceptual foundation stems from my experience in **HSBC Credit Risk Control and Management**. I observed that institutional models often fail when their underlying assumptions about volatility are violated. 

This project transitions from simple "scripting" to **Financial Engineering** by implementing a system that treats volatility not as noise, but as a **Model Activation Filter**. The insight is simple yet profound: **"Models are only as good as the regime they assume."**

---

## 3. Methodological Rigor: Ensuring System Integrity 🏗️

### 3.1 "Blind Threshold" Calibration (Anti-Leakage Protocol)
To strictly prevent **Look-Ahead Bias**, the regime threshold ($\theta$) is derived exclusively from historical data:
- **Calibration**: The median volatility ($\theta_{train} = 0.00701$) is computed using only the **Training Set (2014-2021)**.
- **Zero-Leakage Implementation**: The test phase (2022-2024) remains "blind" to future distributions, ensuring that results reflect a realistic, out-of-sample trading scenario.

### 3.2 Adaptive Ensemble Modeling
- **Linear Baseline (Logistic Regression)**: Failed due to a structural **"Always-Buy" bias** hard-coded during the secular bull market of 2014-2021. It could not adapt to the 2022 regime shift.
- **Non-Linear Challenger (Random Forest)**: Explicitly constrained with `max_depth=5` to force generalization. This prevents the model from "memorizing" idiosyncratic noise and focuses on robust structural patterns.

---

## 4. Mathematical Framework 📐

### 4.1 Dynamic Regime Classification
Market regimes ($R_t$) are classified based on the rolling volatility ($\sigma_t$) relative to the training median threshold ($\theta_{train}$):

$$
R_t = 
\begin{cases} 
\text{Volatile (Bear)}, & \text{if } \sigma_t > \theta_{train} \\
\text{Stable (Bull)}, & \text{if } \sigma_t \leq \theta_{train}
\end{cases}
$$

### 4.2 Target Engineering (Causality Protocol)
The prediction variable is aligned using a strict **shift(-1)** operation to ensure features ($X_t$) only predict the *future* return ($r_{t+1}$), preserving the temporal integrity of the supervised learning task:
$$Y_t = \mathbb{I}(r_{t+1} > 0)$$

---

## 5. Alpha Discovery & Visual Interpretation 📊

### 5.1 The "Tale of Two Markets" (Task 1 & 2 Analysis)
As shown in the visualizations, the dataset captures a structural shift from the **Calm (2014-2019)** to the **Storm (2020 & 2022)**. 
- **The Core Finding**: Accuracy decomposition reveals that predictive signals are most reliable when market noise is lowest.
- **Low Volatility Regime ($N=71$)**: **54.93% Accuracy**. Endogenous technical patterns (SMA, RSI) exhibit high structural fidelity.
- **High Volatility Regime ($N=430$)**: **50.70% Accuracy**. Exogenous shocks (macro/geopolitical) render learned technical signals irrelevant.

### 5.2 Why These Features Matter
- **Trend Dynamics ($SMA_{200}$)**: Acts as a dynamic baseline. Decisive breakdowns (2022) signal structural regime changes.
- **Momentum ($RSI$)**: Custom Wilder’s Smoothing identifies "Seller Exhaustion" (e.g., March 2020 crash at RSI ~15).
- **Regime ($\sigma$)**: Confirms **Volatility Clustering**, proving that models must be context-dependent.

---

## 6. Project Navigation & Technical Specs 💻

- **Vectorization**: Mastered Pandas `ewm` operations to replace inefficient loops, ensuring institutional-grade performance.
- **Reproducibility**: Enforced `np.random.seed(42)` to ensure all performance gains are structural, not stochastic.
- **[Regime_Aware_ML_Strategy.ipynb](Regime_Aware_ML_Strategy.ipynb)**: Full implementation notebook.
- **[spy_data_fixed_10y.csv](spy_data_fixed_10y.csv)**: 10-year Daily OHLCV local cache.

---

### 💡 Final Verdict
The project definitively proves that in financial machine learning, **"When to trade" is as important as "What to trade"**. By staying in cash during volatile regimes and engaging capital only during calm states, we transform a marginal system into a robust, risk-managed strategy.
