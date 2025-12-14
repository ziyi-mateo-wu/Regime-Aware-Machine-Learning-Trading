# Regime-Aware Algorithmic Trading System 📈

> A machine learning-based trading system that dynamically adapts to market regimes (Stable vs. Volatile) to outperform buy-and-hold strategies.

![Status](https://img.shields.io/badge/Status-Completed-success)
![Language](https://img.shields.io/badge/Language-Python-blue)
![Tech](https://img.shields.io/badge/Tech-Scikit%20Learn%20%7C%20Pandas-orange)

### 📌 Project Overview
This project implements an algorithmic trading strategy utilizing **Regime-Switching mechanisms**. By classifying market states using **Statistical Volatility Thresholding**, the system deploys adaptive models to optimize risk-adjusted returns.

### 🚀 Key Features
* **Regime Detection:** Dynamically classifies market states into *Stable* or *Volatile* using rolling median volatility.
* **Adaptive Strategy:**
    * **Stable Regime:** Deploys **Random Forest** to capture trend continuations.
    * **Volatile Regime:** Deploys **Gradient Boosting** to identify non-linear reversal patterns.
* **Risk Management:** Strictly enforces **look-ahead bias prevention** during feature engineering.

### 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `scikit-learn`, `pandas`, `yfinance`, `matplotlib`, `numpy`
* **Techniques:** Ensemble Learning, Vectorised Backtesting

### 📊 Detailed Analysis (Interactive)

The following section contains the key visualizations generated from the model, providing deep insights into market dynamics and model performance.

<details>
<summary><strong>🔻 Click here to expand Visualizations & Charts</strong></summary>
<br>

#### 1. SPY 10-Year Historical Price (2014–2024)
This chart visualizes the SPY ETF’s price trajectory over the past decade, highlighting a key **structural break** during the COVID-19 crash in early 2020. The dataset contains **2,516 clean trading days** and is locally cached to ensure reproducibility and eliminate API dependency.

<img width="100%" alt="SPY Price History" src="https://github.com/user-attachments/assets/13b2d7ff-f5fd-4c57-a232-d58ef43adfba" />

#### 2. Analytical Observations & Theoretical Justification (RSI Analysis)
The feature‑engineering analysis reveals three complementary dimensions of market behavior essential for regime‑aware modeling:

* **Trend Dynamics:** Captured through the **200‑day SMA**, showing that from 2014–2019 the long‑term average acted as reliable support, whereas the decisive breakdown in 2022 marked a structural regime shift.
* **Sentiment (RSI):** Quantified via Wilder’s EWM smoothing, highlighting psychological extremes—most notably the March 2020 liquidity crisis where RSI collapsed to **~15**, signaling forced liquidation and setting the stage for sharp **mean‑reversion**.
* **Volatility Clustering:** Empirically confirming the **heteroscedastic nature** of financial time series, with long calm periods punctuated by explosive spikes in 2020 and 2022.

These engineered features form the foundation of the regime‑aware framework and motivate the transition to constructing a prediction target that avoids look‑ahead bias.

<img width="100%" alt="RSI and Volatility Analysis" src="https://github.com/user-attachments/assets/7f3ef468-8156-4c18-b522-23fb3eaba6dd" />

#### 3. Model Breakdown vs Statistical Edge: A Regime-Aware Comparison
The confusion matrices expose a regime-dependent modeling flaw in linear models versus non-linear ensemble methods:

* **Logistic Regression Failure:** Achieved only **49.50% accuracy** and predicted “Up” on every test day. It failed to generalize beyond the bull regime, reflecting an “Always Buy” bias hard-coded by linear assumptions.
* **Random Forest Edge:** Achieved **51.30% accuracy** and correctly identified **26 Down days**. By leveraging feature interactions (e.g., Volatility × RSI), it captured latent non-linear signals.

This **1.80% edge** above random chance validates the regime-aware hypothesis. Future iterations will further dissect model performance across specific volatility-defined regimes to isolate Alpha persistence.

<img width="100%" alt="Confusion Matrix Comparison" src="https://github.com/user-attachments/assets/7bc21584-54a1-4dcf-9a65-a09d2765bf7f" />

<br>
<p align="center">
  <a href="trading_strategy.ipynb">
    <img src="https://img.shields.io/badge/View_Source_Code-.ipynb_File-blue?style=for-the-badge&logo=jupyter" alt="View Code">
  </a>
</p>

</details>

### 💻 How to Run
1.  Clone the repository.
2.  Install dependencies: `pip install pandas scikit-learn yfinance matplotlib`
3.  Run the script: `python trading_strategy.py`
