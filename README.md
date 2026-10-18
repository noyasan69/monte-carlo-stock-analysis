# 📈 Monte Carlo Stock Price Prediction & Risk Analysis

An institutional-grade Monte Carlo simulation and financial risk modeling engine built with Python, NumPy, Pandas, Matplotlib, and Plotly.

---

## 🚀 Key Features

- **Geometric Brownian Motion (GBM) Engine**:
  Vectorized simulation implementing Itô's calculus drift correction ($\mu - \frac{1}{2}\sigma^2$) across $N$ simulated paths and multiple forecast horizons (30, 90, 252 trading days).
- **Comprehensive Downside Risk Profile**:
  - **Value at Risk (VaR)** at 95% confidence (maximum expected loss).
  - **Conditional Value at Risk (CVaR / Expected Shortfall)** measuring the average loss in the tail distribution beyond VaR.
  - Probability of profit vs. loss, confidence intervals, median vs. mean trajectories.
- **Dark-Themed Visualizations**:
  - Historical Price Trend & Daily Log Returns with mean drift.
  - Return Distribution histogram fitted against Gaussian density + Shapiro-Wilk normality test.
  - Fan Chart with 5th, 25th, 50th (median), 75th, and 95th percentile cones.
  - Terminal Price Distributions across forecast horizons with Gaussian Kernel Density Estimation (KDE).
  - VaR & CVaR Tail Loss Dashboards highlighting extreme drawdowns in red.
- **Interactive Plotly Visualization**:
  Exported HTML interactive fan chart supporting zoom, pan, hover inspection, and individual path trajectories.
- **Bonus Multi-Asset Portfolio Simulation**:
  Calculates portfolio return $\mu_{port}$, portfolio variance $w^T \Sigma w$, annualized Sharpe ratio, and displays an asset cross-correlation matrix heatmap.

---

## 🛠️ Quickstart

### 1. Installation
Install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Run the Script
```bash
# Interactive mode (prompts for ticker):
python monte_carlo.py

# Command-line mode:
python monte_carlo.py --ticker AAPL --sims 10000 --years 5 --confidence 0.95
```

### 3. Google Colab / Jupyter Notebook
Open and run [`monte_carlo_stock.ipynb`](file:///monte_carlo_stock.ipynb) directly in Google Colab or VS Code.

---

## 📁 Generated Outputs

When executed, charts and data are saved into the `outputs/` directory:
- `01_historical.png`: Price history and log returns.
- `02_return_dist.png`: Normality fit & 1-day VaR threshold.
- `03_fan_30d.png`, `03_fan_90d.png`, `03_fan_252d.png`: Monte Carlo fan charts.
- `04_final_dist.png`: Final price probability densities.
- `05_var_cvar.png`: VaR & CVaR tail loss dashboards.
- `06_correlation.png`: Portfolio correlation heatmap.
- `<TICKER>_interactive_fan.html`: Standalone interactive Plotly fan chart.
- `<TICKER>_monte_carlo_results.csv`: Tabular risk summary across all horizons.
- `<TICKER>_sim_finals_<H>d.csv`: Raw simulated terminal prices.

---

## 📐 Mathematical Formulation

### 1. Log Returns
$$\ln\left(\frac{P_t}{P_{t-1}}\right)$$

### 2. Geometric Brownian Motion (GBM)
$$S(t+\Delta t) = S(t) \cdot \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z\right)$$
where $Z \sim \mathcal{N}(0, 1)$ and $\left(\mu - \frac{1}{2}\sigma^2\right)$ is the Itô drift correction.

### 3. Value at Risk (VaR) & CVaR
$$\text{VaR}_\alpha = \text{Percentile}(R, 1 - \alpha)$$
$$\text{CVaR}_\alpha = \mathbb{E}\left[R \mid R \le \text{VaR}_\alpha\right]$$

---

*Disclaimer: For educational and quantitative research purposes only. Not financial advice.*
