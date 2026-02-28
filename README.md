# Regime-Aware Machine Learning System for Market Strategy Prediction 📈

> [cite_start]**Quantitative Research Framework**: Investigating how heteroscedastic volatility regimes shape the predictive boundaries of machine learning models in non-stationary financial markets[cite: 3, 29, 34].

[![Status](https://img.shields.io/badge/Strategy-Volatility%20Filtering-blueviolet?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Performance](https://img.shields.io/badge/Calm%20Regime%20Acc-54.93%25-green?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Language](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)

---

## 1. Executive Summary 📄

[cite_start]Financial time-series forecasting is inherently challenging due to the **non-stationary** and **stochastic** nature of markets[cite: 29]. [cite_start]Traditional ML models often fail because they assume a static environment and ignore **heteroscedasticity**—the clustering of volatility[cite: 30, 31, 398].

[cite_start]This project proposes a **"Regime-Aware"** framework that[cite: 33]:
- [cite_start]**Quantifies Market States**: Segments data into "Calm" and "Volatile" regimes using a statistically grounded threshold[cite: 42, 729].
- [cite_start]**Discovers the Alpha Boundary**: Proves that model accuracy soars to **54.93%** in stable conditions but degrades to near-random (50.70%) during shocks[cite: 882, 883, 884].
- [cite_start]**Engineers a "Fair Weather Alpha" Strategy**: Transforms a weak 51.30% classifier into a robust, regime-conditional system by identifying *when* to trade[cite: 891, 910, 911].

---

## 2. Context & Motivation: From Industry to Theory 💡

[cite_start]The conceptual foundation stems from my experience in **HSBC Credit Risk Control and Management**[cite: 19, 914]. [cite_start]I observed that institutional models often fail when their underlying assumptions about volatility are violated[cite: 915]. 

[cite_start]This project transitions from simple "scripting" to **Financial Engineering** by implementing a system that treats volatility not as noise, but as a **Model Activation Filter**[cite: 917, 919].

---

## 3. Methodological Rigor: Ensuring System Integrity 🏗️

### 3.1 "Blind Threshold" Calibration (Anti-Leakage)
[cite_start]To strictly prevent **Look-Ahead Bias**, the regime threshold ($\theta$) is derived exclusively from historical data[cite: 415, 730, 925]:
- [cite_start]**Calibration**: The median volatility ($\theta_{train} = 0.00701$) is computed using only the **Training Set (2014-2021)**[cite: 731, 760, 805, 880].
- [cite_start]**Zero-Leakage Implementation**: The test phase (2022-2024) remains "blind" to future distributions, ensuring that results reflect a realistic trading scenario[cite: 522, 523, 598, 758].

### 3.2 Adaptive Ensemble Modeling
- [cite_start]**Linear Baseline (Logistic Regression)**: Failed due to a structural **"Always-Buy" bias** hard-coded during the secular bull market of 2014-2021[cite: 524, 706, 714].
- [cite_start]**Non-Linear Challenger (Random Forest)**: Explicitly constrained with `max_depth=5` to force generalization and prevent the model from memorizing financial noise[cite: 530, 531, 563, 604].

---

## 4. Mathematical Framework 📐

### 4.1 Dynamic Regime Classification
[cite_start]Market regimes ($R_t$) are classified based on the rolling volatility ($\sigma_t$) relative to the training median threshold ($\theta_{train}$)[cite: 735, 736]:

$$
R_t = 
\begin{cases} 
\text{Volatile}, & \text{if } \sigma_t > \theta_{train} \\
\text{Stable}, & \text{if } \sigma_t \leq \theta_{train}
\end{cases}
$$



### 4.2 Target Engineering (Causality Protocol)
[cite_start]The prediction variable is aligned using a strict **shift(-1)** operation to ensure features ($X_t$) only predict the *future* return ($r_{t+1}$)[cite: 419, 420, 436, 497]:
$$Y_t = \mathbb{I}(r_{t+1} > 0)$$

---

## 5. Alpha Discovery & Visual Interpretation 📊

### 5.1 The "Tale of Two Markets" (Task 1 & 2)
[cite_start]As shown in **Figure 6.1**, the dataset captures a structural shift from the **Calm (2014-2019)** to the **Storm (2020 & 2022)**[cite: 206, 207, 208]. [cite_start]This visual evidence of **Volatility Clustering** validates the necessity of a Regime-Aware architecture[cite: 214, 397, 399].

### 5.2 Strategy Resilience (Task 6 Visualization)
[cite_start]**The Core Finding**: Accuracy decomposition reveals that predictive signals are most reliable when market noise is lowest[cite: 918].
* **Low Volatility ($N=71$)**: **54.93% Accuracy**. [cite_start]Technical patterns (SMA, RSI) exhibit high structural fidelity[cite: 882, 889].
* **High Volatility ($N=430$)**: **50.70% Accuracy**. [cite_start]Exogenous shocks (macro/geopolitical) render technical signals irrelevant[cite: 883, 890].

---

## 6. Project Navigation 💻

- [cite_start]**[Regime_Aware_ML_Strategy.ipynb](Regime_Aware_ML_Strategy.ipynb)**: Full implementation, including vectorized RSI via Wilder’s Smoothing and deterministic training with `np.random.seed(42)`[cite: 35, 53, 75, 231, 921, 923].
- [cite_start]**[spy_data_fixed_10y.csv](spy_data_fixed_10y.csv)**: 10-year Daily OHLCV local cache (2,516 observations)[cite: 86, 111, 204, 205].

---

### 💡 Final Verdict
[cite_start]The project definitively proves that **"When to trade" is as important as "What to trade"**[cite: 911]. [cite_start]In financial ML, Alpha is conditional; it thrives when the market behaves "normally" and degrades into randomness during chaos[cite: 891, 900].
