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

### 📊 Detailed Analysis & Code

[![View in nbviewer](https://img.shields.io/badge/View%20Full%20Notebook-nbviewer-orange?style=for-the-badge&logo=jupyter)](你的_nbviewer_新网址_粘贴在这里)

> 👆 **Click the button above** to render the full analysis with interactive charts (High Performance).

[![Preview](results_cover.png)](你的_nbviewer_新网址_粘贴在这里)

![Model Performance](results_plot.png)
### 💻 How to Run
1.  Clone the repository.
2.  Install dependencies: `pip install pandas scikit-learn yfinance matplotlib`
3.  Run the script: `python trading_strategy.py`
