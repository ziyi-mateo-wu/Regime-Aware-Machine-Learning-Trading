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

The following section contains the key visualizations generated from the model. 

<details>
<summary><strong>🔻 Click here to expand Visualizations & Charts</strong></summary>
<br>

#### 1. SPY 10-Year Historical Price (2014–2024)
SPY 10-Year Historical Price
This chart visualizes the SPY ETF’s price trajectory over the past decade, highlighting a key structural break during the COVID-19 crash in early 2020. The dataset () contains 2,516 clean trading days and is locally cached to ensure reproducibility and eliminate API dependency.
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/13b2d7ff-f5fd-4c57-a232-d58ef43adfba" />

#### 2. Regime Classification (Volatility States)
![Regimes](plot2.png)

#### 3. Model Diagnostics
![Diagnostics](plot3.png)

<br>
<p align="center">
  <a href="trading_strategy.ipynb">
    <img src="https://img.shields.io/badge/View_Source_Code-.ipynb_File-blue?style=for-the-badge&logo=jupyter" alt="View Code">
  </a>
</p>

</details>

![Model Performance](results_plot.png)
### 💻 How to Run
1.  Clone the repository.
2.  Install dependencies: `pip install pandas scikit-learn yfinance matplotlib`
3.  Run the script: `python trading_strategy.py`
