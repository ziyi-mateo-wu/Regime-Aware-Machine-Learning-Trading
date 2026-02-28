# Regime-Aware Machine Learning System for Market Strategy Prediction 📈

> A regime-aware quantitative research project on how volatility states shape machine learning performance in financial markets.

[![Focus](https://img.shields.io/badge/Focus-Volatility%20Regimes-blueviolet?style=for-the-badge)](#)
[![Edge](https://img.shields.io/badge/Calm%20Regime%20Acc-54.93%25-green?style=for-the-badge)](#)
[![Language](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)

---

## 1. Executive summary

Financial time series are **non-stationary and heteroscedastic**: long stretches of low volatility are punctuated by sharp, clustered shocks. Standard machine learning models typically treat all observations as if they were drawn from a single, stable distribution, which leads to **performance collapse in stressed regimes**.

This project develops a **Regime-Aware Machine Learning System** that:

- **Segments markets into volatility regimes** using a statistically grounded threshold on rolling volatility.
- **Trains a classifier on technical and volatility features** to predict next-day direction of SPY.
- **Conditions model usage on regime**, turning the model into a **Risk-On / Risk-Off switch** rather than an always-on predictor.

Key empirical finding:

- Overall Random Forest accuracy (2022–2024): **51.30%**  
- Accuracy in **low-volatility regime**: **54.93%**  
- Accuracy in **high-volatility regime**: **50.70%**  

> **Core insight:** The model’s predictive edge is **regime-dependent**. It adds value only in calm markets and should be explicitly disabled in volatile regimes.

---

## 2. Research context & objective

### 2.1 Problem setting

Traditional ML models in finance often:

- Assume **stationarity** of return distributions.
- Ignore **volatility clustering** and structural breaks.
- Evaluate performance on mixed-regime test sets, masking where the model actually works.

This project focuses on:

- **Directional prediction** of SPY (S&P 500 ETF) on a **next-day horizon**.
- Explicitly modeling **volatility regimes** and **conditioning model usage** on those regimes.

### 2.2 Objective

Design and evaluate a **regime-aware ML framework** that:

1. Quantifies the impact of volatility-based regime segmentation on predictive accuracy.
2. Identifies whether a model’s edge is concentrated in specific volatility states.
3. Converts a marginal classifier into a **regime-conditional trading component**.

---

## 3. Data & feature engineering

### 3.1 Data universe

- **Asset:** SPY (S&P 500 ETF)  
- **Frequency:** Daily OHLCV  
- **Period:** 2014-01-01 to 2024-01-01  
- **Observations:** 2,516 trading days (post-cleaning)

A **local CSV cache** is implemented to:

- Avoid API instability and rate limits.
- Ensure reproducibility across runs and environments.

### 3.2 Core features

The feature set is designed to capture **trend**, **momentum**, and **regime**:

1. **Log returns**  
   

\[
   r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
   \]



2. **Trend – Simple Moving Averages (SMA)**  
   - \( \text{SMA}_{50} \): medium-term trend  
   - \( \text{SMA}_{200} \): long-term structural trend  

3. **Momentum – RSI (Wilder’s Smoothing)**  
   A **custom RSI implementation** using Exponential Weighted Moving Average (EWMA) to match industry-standard platforms.

4. **Regime indicator – Rolling volatility**  
   - 20-day rolling standard deviation of log returns:
   

\[
   \sigma_t = \sqrt{\frac{1}{N-1}\sum_{i=0}^{N-1}(r_{t-i} - \bar{r})^2}
   \]


   - Serves as a quantitative proxy for **market “fear”** and stress.

---

## 4. Methodology & modeling pipeline

### 4.1 Target definition & anti-leakage protocol

The prediction task is framed as **binary classification**:

- Horizon: **next-day** return \( r_{t+1} \)
- Target:
  

\[
  Y_t = \mathbb{I}(r_{t+1} > 0)
  \]



To strictly prevent **Look-Ahead Bias**:

- Features at time \( t \) are aligned only with \( r_{t+1} \).
- The final row (with undefined \( r_{t+1} \)) is dropped.
- No future information is used in feature construction.

### 4.2 Chronological train–test split

To respect time ordering:

- **Training set:** 2014-10-16 to 2021-12-31  
- **Test set:** 2022-01-03 to 2023-12-29  

No shuffling is applied. This simulates a realistic **out-of-sample** evaluation across a structurally different regime (bear market + recovery).

### 4.3 Models

Two models are compared:

- **Baseline:** Logistic Regression  
  - Linear decision boundary  
  - Tends to collapse into an “always buy” bias.

- **Challenger:** Random Forest Classifier  
  - Non-linear ensemble  
  - Captures interactions between volatility, trend, and momentum.

---

## 5. Regime-aware framework

### 5.1 Regime classification

Volatility regimes are defined using the **training-set median** of rolling volatility:

- Let \( \theta_{\text{train}} \) be the median of \( \sigma_t \) on the training set.  
- Regime assignment:
  

\[
  R_t =
  \begin{cases}
  \text{Volatile (High Vol)}, & \sigma_t > \theta_{\text{train}} \\
  \text{Calm (Low Vol)}, & \sigma_t \leq \theta_{\text{train}}
  \end{cases}
  \]



This is a **“blind” threshold**:  
- Calibrated **only** on training data.  
- Applied unchanged to the test period (2022–2024).

### 5.2 “Fair Weather Alpha” interpretation

The model is not treated as a static, always-on predictor. Instead:

- In **high volatility**, technical signals are overwhelmed by exogenous shocks → model edge disappears.
- In **low volatility**, technical structure is more stable → model edge emerges.

The ML system is therefore interpreted as a **regime-conditional component**:

- **Risk-Off:** High volatility → stay in cash.  
- **Risk-On:** Low volatility → engage capital using model signals.

---

## 6. Results

### 6.1 Model-level performance (2022–2024 test set)

| Metric              | Logistic Regression (Baseline) | Random Forest (Challenger) |
|---------------------|---------------------------------|----------------------------|
| Overall Accuracy    | 49.50%                          | **51.30%**                 |
| Behaviour           | “Always Buy” bias               | Non-linear signal capture  |

The overall edge is modest—but this masks a **regime-dependent structure**.

### 6.2 Regime-level performance (Random Forest)

| Regime          | Volatility Condition          | Accuracy   | Sample Size |
|-----------------|------------------------------|-----------:|------------:|
| **Low Vol**     | \( \sigma_t \leq \theta_{\text{train}} \) | **54.93%** | 71          |
| **High Vol**    | \( \sigma_t > \theta_{\text{train}} \)    | 50.70%     | 430         |

- **Performance spread:** +4.23% in calm regimes vs. high-vol regimes.  
- The model’s edge is **concentrated in low-volatility states**.

> **Strategic conclusion:** The classifier is best deployed as a **volatility filter**—a conditional trading switch, not a universal predictor.

---

## 7. Visual analysis

*(Images referenced below are stored in the repository as pre-rendered outputs.)*

### 7.1 SPY 10-year structural context

- Visualizes SPY price from 2014–2024.
- Highlights:
  - 2020 COVID-19 crash  
  - 2022 structural regime shift  

This motivates the need for **regime-aware modeling**.

### 7.2 Feature diagnostics

Three-panel visualization:

1. **Trend:** Price vs. SMA\(_{50}\) and SMA\(_{200}\)  
2. **Momentum:** RSI with overbought/oversold bands  
3. **Regime:** Rolling volatility with shaded intensity  

Key observations:

- SMA\(_{200}\) acts as **dynamic support** in 2014–2019, then fails in 2022 → structural break.  
- RSI extremes (e.g., ~15 in March 2020) indicate **seller exhaustion** and precede sharp mean reversion.  
- Volatility exhibits clear **clustering**, confirming heteroscedasticity.

### 7.3 Regime performance & confusion matrices

- Bar charts show **accuracy by regime**, visualizing the “Fair Weather Alpha”.  
- Confusion matrices compare:
  - Logistic Regression’s “always buy” tendency.  
  - Random Forest’s more balanced classification.

---

## 8. Strategic implications

This project supports several practical conclusions:

1. **“When to trade” is as important as “what to trade.”**  
   A modestly predictive model can be materially improved by **turning it off** in the wrong regime.

2. **Volatility is a first-class input, not just a risk metric.**  
   Volatility should be treated as a **conditioning variable** for model deployment.

3. **Regime-aware ML is a natural bridge between risk management and alpha generation.**  
   The system behaves like a **Risk-On / Risk-Off overlay**, consistent with institutional risk practices.

---

## 9. Limitations & future work

### 9.1 Limitations

- **Single asset:** Only SPY is analyzed; no cross-asset validation.  
- **Limited feature set:** No macro, cross-sectional, or options-implied features.  
- **Simple models:** Only Logistic Regression and Random Forest are tested.  
- **Sample imbalance:** Low-volatility regime has relatively fewer observations (N=71).

### 9.2 Future extensions

Potential directions:

- Extend to **multi-asset universes** (e.g., sector ETFs, global indices).  
- Incorporate **macro variables** and **options-implied volatility** (e.g., VIX).  
- Explore **sequence models** (LSTM, Temporal CNN, Transformers) with regime conditioning.  
- Integrate **transaction costs** and **position sizing** into a full backtest.

---

## 10. Repository structure & usage

### 10.1 Repository layout

```text
Regime-Aware-Machine-Learning-Trading/
├── Regime_Aware_ML_Strategy.ipynb   # Main research notebook (fully annotated)
├── spy_data_fixed_10y.csv           # Local cache of SPY OHLCV data (10 years)
├── README.md                        # Project documentation (this file)
└── (optional) figures/              # Exported plots and visualizations
