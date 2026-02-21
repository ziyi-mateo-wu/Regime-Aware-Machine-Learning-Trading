# Regime-Aware Machine Learning Strategy 📈

> A rigorous quantitative framework analyzing how volatility regimes impact machine learning efficacy in financial markets.

[![Status](https://img.shields.io/badge/Strategy-Volatility%20Filtering-blueviolet?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Performance](https://img.shields.io/badge/Calm%20Regime%20Acc-54.93%25-green?style=for-the-badge)](https://github.com/ziyi-mateo-wu)
[![Language](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)

### 💡 Context & Inspiration
The conceptual foundation for this project stems from practices in **Credit Risk Control (HSBC)**. Traditional models often fail when their underlying assumptions about market stability are violated.

This project implements a **"Fair Weather Alpha"** strategy: instead of treating prediction as a static task, it builds a system that strictly filters for the "regime" it assumes—engaging capital only when market noise is low and technical signals are reliable.

---

### 🏗️ Project Architecture
The pipeline is structured into four distinct engineering phases to ensuring statistical rigor and data integrity.

#### Phase 1: Quantitative Feature Engineering
Derived technical indicators to capture market dynamics:
* **Trend:** 200-day Simple Moving Average (SMA) to identify long-term support.
* **Momentum:** Relative Strength Index (RSI) with **Wilder’s Smoothing** to detect liquidity crises (e.g., March 2020).
* **Volatility:** Rolling standard deviation of log-returns to quantify market fear.

#### Phase 2: "Blind" Regime Calibration (Anti-Leakage)
To strictly prevent **Look-Ahead Bias**, the regime threshold was calibrated exclusively on historical data:
* **Calibration:** Calculated the median volatility ($\sigma_{train} \approx 0.007$) solely from the Training Set (2014-2021).
* **Application:** Applied this *frozen* threshold to the Test Set (2022-2024) to segment markets into **"Calm"** vs. **"Volatile"**.

#### Phase 3: Adaptive Ensemble Modeling
Comparative analysis of linear vs. non-linear estimators:
* **Baseline:** Logistic Regression (Linear) — Failed to capture complex patterns (Accuracy: ~49%).
* **Challenger:** Random Forest (Non-Linear Ensemble) — Successfully captured latent interactions between Volatility and RSI.

---

### 📐 Mathematical Framework
The strategy relies on rigorous quantitative definitions for signal generation.

**1. Dynamic Regime Classification**
Market regimes ($R_t$) are classified based on the rolling volatility ($\sigma_t$) relative to the historical median threshold ($\theta_{train}$):

$$
R_t = 
\begin{cases} 
\text{Volatile (Bear)}, & \text{if } \sigma_t > \theta_{train} \\
\text{Stable (Bull)}, & \text{if } \sigma_t \leq \theta_{train}
\end{cases}
$$

**2. Target Engineering**
To prevent look-ahead bias, the target $Y_t$ is aligned with the *next day's* return $r_{t+1}$:
$$Y_t = \mathbb{I}(r_{t+1} > 0)$$

---

### 📊 Performance & Alpha Discovery

The analysis revealed a crucial divergence in model performance, validating the **"Fair Weather Alpha"** hypothesis.

#### 1. Model Comparison (Test Set 2022-2024)

| Metric | Logistic Regression (Baseline) | Random Forest (Challenger) |
| :--- | :---: | :---: |
| **Overall Accuracy** | 49.50% (Fail) | **51.30%** (Edge) |
| **Behavior** | "Always Buy" Bias | Captured Non-linear Signals |

#### 2. The Regime Spread (Key Discovery)
Dissecting the Random Forest performance by volatility state reveals the true source of Alpha:

| Regime | Volatility State | Accuracy | Sample Size |
| :--- | :--- | :---: | :---: |
| **Low Volatility** | $\sigma_t \le 0.007$ | **54.93%** PASS | $N=71$ |
| **High Volatility** | $\sigma_t > 0.007$ | 50.70% FAILED | $N=430$ |

> **Strategic Insight:** The model possesses a **+4.23% Performance Spread** in calm markets. In high volatility, exogenous shocks render technical signals irrelevant.

---

### 📉 Detailed Analysis & Visualizations



#### 1. SPY 10-Year Historical Price (Structural Breaks)
Visualizing the 2020 COVID-19 crash and the 2022 regime shift. The dataset contains **2,516 clean trading days**.
<img width="100%" alt="SPY Price" src="https://github.com/user-attachments/assets/3805d9b4-6b7d-42e5-9cb7-abaf01de1201" />

#### 2. Analytical Observations: Why These Features Matter?
*(Visualizing Trend, Momentum, and Regime components)*
* **Trend Dynamics ($SMA_{200}$):** The 200-day SMA acted as reliable **"Dynamic Support"** from 2014-2019. The decisive breakdown in 2022 signaled a structural regime change, validating SMAs as effective filters for Bear Markets.
* **Sentiment Extremes ($RSI$):** During the March 2020 liquidity crisis, RSI plunged to **~15** (far below the standard 30). This mathematical extreme signaled **"Seller Exhaustion"**, preceding a sharp mean-reversion event.
* **Volatility Clustering (Regime):** The data exhibits distinct **"Heteroscedasticity"** (periods of calm vs. explosions of risk). This visually proves that a static model would fail during the 2020/2022 shocks, necessitating our **Regime-Aware architecture**.

<img width="100%" alt="Feature Engineering Analysis" src="https://github.com/user-attachments/assets/4f2b87e5-320f-4fe8-a667-9ab00e430ada" />

#### 3. Regime-Based Performance Analysis
The bar chart below illustrates the "Fair Weather Alpha". Note the clear performance degradation in the High Volatility regime (Red bar) compared to the Low Volatility regime (Green bar).



<img width="100%" alt="Regime Analysis" src="regime_analysis.png" />

#### 4. Confusion Matrix Comparison
Contrasting the "Always Buy" bias of Logistic Regression against the nuanced predictions of Random Forest.
<img width="100%" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/a85f0d3b-c562-40c9-b50a-9cc7decb01b1" />

<p align="center">
  <a href="Regime_Aware_ML_Strategy.ipynb">
    <img src="https://img.shields.io/badge/View_Source_Code-.ipynb_File-blue?style=for-the-badge&logo=jupyter" alt="View Code">
  </a>
</p>

---

### 💡 Strategic Implication: The Volatility Filter

The project proves that **"When to trade" is as important as "What to trade."** Instead of a static "always-on" predictor, the model functions best as a **Risk-On / Risk-Off Switch**:

1. **Risk-Off (Cash):** When Rolling Volatility > 0.007, the model's edge disappears. **Strategy: Stay in Cash.**
2. **Risk-On (Trade):** When Rolling Volatility ≤ 0.007, the model achieves **~55% accuracy**. **Strategy: Engage Capital.**

This transformation turns a marginal 51% model into a robust, regime-conditional trading system.

---

### 💻 How to Run

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn yfinance matplotlib seaborn
