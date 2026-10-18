# ============================================================
#   MONTE CARLO STOCK PRICE PREDICTION & RISK ANALYSIS
#   Google Colab & Local Python — Full Project
#   Author  : Aryan (Enhanced Edition)
#   Version : 1.1
# ============================================================
# HOW TO USE:
#   1. Run from CLI: python monte_carlo.py --ticker AAPL
#      Or simply:    python monte_carlo.py (prompts for ticker)
#   2. On Colab: Paste this code or upload and run.
# ============================================================

import importlib.util
import subprocess
import sys
import os
import argparse
import warnings
from datetime import datetime, timedelta

# Fix Windows console UTF-8 emoji encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Dependency Auto-Check ────────────────────────────────────
REQUIRED_PACKAGES = ("yfinance", "plotly", "scipy", "numpy", "pandas", "matplotlib")
for _pkg in REQUIRED_PACKAGES:
    if importlib.util.find_spec(_pkg) is None:
        print(f"📦 Installing missing dependency: {_pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg, "--quiet"])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy import stats
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Global plot style (dark theme) ───────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "#0d0d0d",
    "axes.facecolor"   : "#141414",
    "axes.edgecolor"   : "#333333",
    "axes.labelcolor"  : "#cccccc",
    "text.color"       : "#dddddd",
    "xtick.color"      : "#aaaaaa",
    "ytick.color"      : "#aaaaaa",
    "grid.color"       : "#222222",
    "grid.linestyle"   : "--",
    "grid.linewidth"   : 0.5,
    "lines.linewidth"  : 1.1,
    "font.family"      : "sans-serif",
    "legend.facecolor" : "#1a1a1a",
    "legend.edgecolor" : "#444444",
})

GOLD   = "#FFD700"
GREEN  = "#00FF88"
RED    = "#FF4C4C"
BLUE   = "#4CA3FF"
PURPLE = "#C084FC"
ORANGE = "#FF9F45"
WHITE  = "#F0F0F0"

print("✅  Libraries and styling ready.\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2 — Configuration: User Inputs                    ║
# ╚══════════════════════════════════════════════════════════╝

def parse_arguments():
    parser = argparse.ArgumentParser(description="Monte Carlo Stock Price & Risk Simulation")
    parser.add_argument("--ticker", type=str, default=None, help="Stock ticker (e.g. AAPL, TSLA, RELIANCE.NS)")
    parser.add_argument("--years", type=int, default=5, help="Years of historical data (default: 5)")
    parser.add_argument("--sims", type=int, default=10_000, help="Number of simulations (default: 10,000)")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save plots and CSVs")
    args, unknown = parser.parse_known_args()
    return args

args = parse_arguments()

if args.ticker:
    TICKER = args.ticker.strip().upper()
else:
    raw_input = input("Enter stock ticker (e.g. AAPL, TSLA, RELIANCE.NS): ").strip().upper()
    TICKER = raw_input if raw_input else "AAPL"

HIST_YEARS       = args.years
N_SIMULATIONS    = args.sims
FORECAST_DAYS    = [30, 90, 252]   # Prediction horizons
CONFIDENCE_LEVEL = args.confidence
RANDOM_SEED      = args.seed

np.random.seed(RANDOM_SEED)

# Save location
if args.output_dir:
    OUTPUT_DIR = args.output_dir
elif os.path.isdir("/content"):
    OUTPUT_DIR = "/content/outputs"
else:
    OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n📌  Ticker          : {TICKER}")
print(f"📅  History         : {HIST_YEARS} years")
print(f"🔁  Simulations     : {N_SIMULATIONS:,}")
print(f"🔭  Horizons (days) : {FORECAST_DAYS}")
print(f"📊  Confidence      : {round(CONFIDENCE_LEVEL*100)}%")
print(f"📁  Output Dir      : {OUTPUT_DIR}\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3 — Download & Validate Historical Data           ║
# ╚══════════════════════════════════════════════════════════╝

def download_stock_data(ticker: str, years: int) -> pd.DataFrame:
    """Download adjusted-close price history from Yahoo Finance."""
    end   = datetime.today()
    start = end - timedelta(days=int(years * 365.25))

    print(f"⬇️   Downloading {ticker} …", end=" ", flush=True)
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as e:
        sys.exit(f"\n❌  Network error: {e}")

    if raw is None or raw.empty:
        sys.exit(f"\n❌  No data returned for '{ticker}'. "
                 "Check the ticker symbol and try again.")

    # Flatten MultiIndex if present in recent yfinance releases
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.levels[0]:
            df = raw["Close"].copy()
            if isinstance(df, pd.DataFrame):
                df.columns = ["Close"] if df.shape[1] == 1 else df.columns
            else:
                df = df.to_frame(name="Close")
        else:
            raw.columns = raw.columns.get_level_values(0)
            df = raw[["Close"]].copy()
    else:
        df = raw[["Close"]].copy()

    df.dropna(inplace=True)

    if df.empty:
        sys.exit(f"\n❌  No valid price rows found for '{ticker}'.")

    missing_pct = raw["Close"].isna().mean() * 100 if "Close" in raw else 0
    if isinstance(missing_pct, pd.Series):
        missing_pct = missing_pct.iloc[0]
    if missing_pct > 10:
        print(f"\n⚠️   Warning: {missing_pct:.1f}% missing values detected.")

    print("done ✓")
    print(f"    Period  : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"    Rows    : {len(df):,} trading days")
    print(f"    Latest  : ₹/$ {float(df['Close'].iloc[-1]):,.2f}\n")
    return df


df = download_stock_data(TICKER, HIST_YEARS)
close = df["Close"].squeeze()


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4 — Feature Engineering                           ║
# ╚══════════════════════════════════════════════════════════╝

log_returns = np.log(close / close.shift(1)).dropna()

# Key statistics
mu_daily  = float(log_returns.mean())          # Daily mean return (drift)
sig_daily = float(log_returns.std(ddof=1))     # Daily volatility (σ)

# Annualised figures (252 trading days / year)
mu_annual  = mu_daily  * 252
sig_annual = sig_daily * np.sqrt(252)

# GBM expects the ARITHMETIC drift; the mean of log-returns already has
# the -σ²/2 term baked in, so add it back before simulating.
mu_gbm = mu_daily + 0.5 * sig_daily ** 2

S0 = float(close.iloc[-1])   # Last known price (starting point)

print("═" * 48)
print("  STATISTICAL PARAMETERS")
print("═" * 48)
print(f"  Last Price (S₀)      : {S0:>12,.4f}")
print(f"  Daily Mean  (μ)      : {mu_daily:>12.6f}")
print(f"  Daily Vol   (σ)      : {sig_daily:>12.6f}")
print(f"  Annual Return        : {mu_annual*100:>11.2f} %")
print(f"  Annual Volatility    : {sig_annual*100:>11.2f} %")
print(f"  Skewness             : {float(log_returns.skew()):>12.4f}")
print(f"  Kurtosis (excess)    : {float(log_returns.kurt()):>12.4f}")
print("═" * 48 + "\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5 — Monte Carlo Engine (Geometric Brownian Motion)║
# ╚══════════════════════════════════════════════════════════╝

def run_monte_carlo(S0: float,
                    mu: float,
                    sigma: float,
                    n_days: int,
                    n_sims: int) -> np.ndarray:
    """
    Vectorised GBM Monte Carlo simulation.

    Returns
    -------
    paths : ndarray, shape (n_days + 1, n_sims)
        Row 0 is S0 for every simulation.
    """
    dt    = 1.0                       # 1 trading day
    drift = (mu - 0.5 * sigma ** 2) * dt

    # Draw ALL random shocks at once  →  shape (n_days, n_sims)
    Z = np.random.standard_normal((n_days, n_sims))

    # Daily log-return for each step & simulation
    daily_log_ret = drift + sigma * np.sqrt(dt) * Z   # (n_days, n_sims)

    # Cumulative product via cumsum on log scale, then exponentiate
    log_price_change = np.vstack([
        np.zeros((1, n_sims)),
        np.cumsum(daily_log_ret, axis=0)
    ])                                                 # (n_days+1, n_sims)

    paths = S0 * np.exp(log_price_change)              # (n_days+1, n_sims)
    return paths


MAX_DAYS = max(FORECAST_DAYS)

print(f"🔁  Running {N_SIMULATIONS:,} simulations over {MAX_DAYS} days …", end=" ", flush=True)
all_paths = run_monte_carlo(S0, mu_gbm, sig_daily, MAX_DAYS, N_SIMULATIONS)
print("done ✓\n")
print(f"    Simulation matrix shape : {all_paths.shape}  "
      f"(days+1 × simulations)\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6 — Risk Metrics Calculator                       ║
# ╚══════════════════════════════════════════════════════════╝

def compute_risk_metrics(paths: np.ndarray,
                         S0: float,
                         horizon: int,
                         conf: float = 0.95) -> dict:
    """
    Compute a full risk profile for a given horizon.
    """
    final_prices   = paths[horizon, :]          # Final price of every sim
    final_returns  = (final_prices - S0) / S0   # Simple return vs S0

    expected_price = float(np.mean(final_prices))
    median_price   = float(np.median(final_prices))
    best_case      = float(np.percentile(final_prices, 95))
    worst_case     = float(np.percentile(final_prices, 5))

    prob_profit = float(np.mean(final_prices > S0) * 100)
    prob_loss   = float(np.mean(final_prices < S0) * 100)

    # Confidence interval on final price
    ci_lo = float(np.percentile(final_prices, (1 - conf) / 2 * 100))
    ci_hi = float(np.percentile(final_prices, (1 + conf) / 2 * 100))

    # Value at Risk (VaR): maximum loss (as % of S0) at confidence level
    VaR = float(np.percentile(final_returns, (1 - conf) * 100))

    # Conditional VaR / Expected Shortfall (CVaR)
    tail_mask = final_returns <= VaR
    CVaR      = float(np.mean(final_returns[tail_mask]))

    return {
        "Horizon (days)"     : horizon,
        "Starting Price"     : round(S0, 4),
        "Expected Price"     : round(expected_price, 4),
        "Median Price"       : round(median_price, 4),
        "Best Case (95th %)" : round(best_case, 4),
        "Worst Case (5th %)": round(worst_case, 4),
        f"CI Lower ({round(conf*100)}%)"  : round(ci_lo, 4),
        f"CI Upper ({round(conf*100)}%)"  : round(ci_hi, 4),
        "Prob. of Profit (%)" : round(prob_profit, 2),
        "Prob. of Loss (%)"   : round(prob_loss, 2),
        f"VaR ({round(conf*100)}%)"       : round(VaR * 100, 4),
        f"CVaR ({round(conf*100)}%)"      : round(CVaR * 100, 4),
    }


results = {}
for h in FORECAST_DAYS:
    results[h] = compute_risk_metrics(all_paths, S0, h, CONFIDENCE_LEVEL)

rows = []
for h, m in results.items():
    rows.append({
        "Horizon"            : f"{h}d",
        "Expected Price"     : m["Expected Price"],
        "Median Price"       : m["Median Price"],
        "Prob Profit %"      : m["Prob. of Profit (%)"],
        "Prob Loss %"        : m["Prob. of Loss (%)"],
        f"VaR {round(CONFIDENCE_LEVEL*100)}%"  : m[f"VaR ({round(CONFIDENCE_LEVEL*100)}%)"],
        f"CVaR {round(CONFIDENCE_LEVEL*100)}%" : m[f"CVaR ({round(CONFIDENCE_LEVEL*100)}%)"],
        "CI Lower"           : m[f"CI Lower ({round(CONFIDENCE_LEVEL*100)}%)"],
        "CI Upper"           : m[f"CI Upper ({round(CONFIDENCE_LEVEL*100)}%)"],
    })

summary_df = pd.DataFrame(rows).set_index("Horizon")
print("═" * 80)
print(f"  RISK SUMMARY — {TICKER}")
print("═" * 80)
print(summary_df.to_string())
print("═" * 80 + "\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 7 — Plot 1: Historical Price + Returns            ║
# ╚══════════════════════════════════════════════════════════╝

fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"hspace": 0.35})
fig.suptitle(f"{TICKER}  —  Historical Price & Daily Returns",
             fontsize=14, color=GOLD, fontweight="bold", y=0.98)

# Panel 1: Price history
ax1 = axes[0]
ax1.plot(close.index, close.values, color=BLUE, linewidth=1.0, label="Close Price")
ax1.fill_between(close.index, close.values, alpha=0.08, color=BLUE)
ax1.set_ylabel("Price", color=WHITE)
ax1.set_title("Adjusted Close Price", color=WHITE, fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True)

# Panel 2: Daily log-returns
ax2 = axes[1]
colors_bar = [GREEN if r >= 0 else RED for r in log_returns.values]
ax2.bar(log_returns.index, log_returns.values, color=colors_bar, width=1, alpha=0.8)
ax2.axhline(0, color=WHITE, linewidth=0.6, linestyle="--")
ax2.axhline(mu_daily, color=GOLD, linewidth=1.0, linestyle="--", label=f"Mean {mu_daily:.5f}")
ax2.set_ylabel("Log Return", color=WHITE)
ax2.set_title("Daily Log Returns", color=WHITE, fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True)

p1_path = os.path.join(OUTPUT_DIR, "01_historical.png")
plt.savefig(p1_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"📊  Plot 1 saved → {p1_path}\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 8 — Plot 2: Return Distribution vs Normal         ║
# ╚══════════════════════════════════════════════════════════╝

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle(f"{TICKER}  —  Return Distribution vs Normal Fit",
             fontsize=13, color=GOLD, fontweight="bold")

n_bins = min(80, int(len(log_returns) ** 0.5 * 4))
ax.hist(log_returns.values, bins=n_bins, density=True,
        color=BLUE, alpha=0.55, label="Observed returns", edgecolor="none")

x_range = np.linspace(log_returns.min(), log_returns.max(), 500)
ax.plot(x_range, stats.norm.pdf(x_range, mu_daily, sig_daily),
        color=GOLD, linewidth=2, label="Normal fit")

var_1d = float(np.percentile(log_returns, (1 - CONFIDENCE_LEVEL) * 100))
ax.axvline(var_1d, color=RED, linewidth=1.5, linestyle="--",
           label=f"1-day VaR ({round(CONFIDENCE_LEVEL*100)}%) = {var_1d:.4f}")

ax.set_xlabel("Daily Log Return", color=WHITE)
ax.set_ylabel("Density", color=WHITE)
ax.legend(fontsize=9)
ax.grid(True)

sample_data = log_returns.sample(min(300, len(log_returns)), random_state=RANDOM_SEED)
_, p_value = stats.shapiro(sample_data)
ax.text(0.02, 0.93,
        f"Shapiro-Wilk p = {p_value:.4f}  "
        f"({'Normal' if p_value > 0.05 else 'Non-normal'} at 5%)",
        transform=ax.transAxes, fontsize=8,
        color=GREEN if p_value > 0.05 else ORANGE,
        bbox=dict(facecolor="#1a1a1a", edgecolor="#444", boxstyle="round,pad=0.3"))

p2_path = os.path.join(OUTPUT_DIR, "02_return_dist.png")
plt.savefig(p2_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"📊  Plot 2 saved → {p2_path}\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 9 — Plot 3: Simulated Price Paths (Fan Chart)     ║
# ╚══════════════════════════════════════════════════════════╝

def plot_simulation_fan(paths: np.ndarray, S0: float, ticker: str, horizon: int, n_display: int = 300):
    days   = np.arange(horizon + 1)
    subset = paths[:horizon + 1, :n_display]

    p5  = np.percentile(paths[:horizon + 1], 5,  axis=1)
    p25 = np.percentile(paths[:horizon + 1], 25, axis=1)
    p50 = np.percentile(paths[:horizon + 1], 50, axis=1)
    p75 = np.percentile(paths[:horizon + 1], 75, axis=1)
    p95 = np.percentile(paths[:horizon + 1], 95, axis=1)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        f"{ticker}  —  Monte Carlo Simulation  |  "
        f"{N_SIMULATIONS:,} paths  |  {horizon}-day horizon",
        fontsize=13, color=GOLD, fontweight="bold")

    ax.plot(days, subset, alpha=0.03, linewidth=0.6, color=BLUE)

    ax.fill_between(days, p5, p95, alpha=0.15, color=BLUE,  label="5th–95th %ile")
    ax.fill_between(days, p25, p75, alpha=0.25, color=BLUE, label="25th–75th %ile")

    ax.plot(days, p50, color=GOLD,   linewidth=2.0, label="Median path")
    ax.plot(days, p95, color=GREEN,  linewidth=1.2, linestyle="--", label="95th %ile")
    ax.plot(days, p5,  color=RED,    linewidth=1.2, linestyle="--", label="5th %ile")
    ax.axhline(S0,     color=WHITE,  linewidth=1.0, linestyle=":",  label=f"S₀ = {S0:,.2f}")

    ax.set_xlabel("Trading Days from Today", color=WHITE)
    ax.set_ylabel("Simulated Price", color=WHITE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)

    fname = os.path.join(OUTPUT_DIR, f"03_fan_{horizon}d.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"📊  Fan chart ({horizon}d) saved → {fname}")


for h in FORECAST_DAYS:
    plot_simulation_fan(all_paths, S0, TICKER, h)
print()


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 10 — Plot 4: Final-Price Histograms               ║
# ╚══════════════════════════════════════════════════════════╝

fig, axes = plt.subplots(1, len(FORECAST_DAYS), figsize=(6 * len(FORECAST_DAYS), 5))
fig.suptitle(f"{TICKER}  —  Distribution of Final Simulated Prices",
             fontsize=13, color=GOLD, fontweight="bold")

if len(FORECAST_DAYS) == 1:
    axes = [axes]

for ax, h in zip(axes, FORECAST_DAYS):
    m       = results[h]
    finals  = all_paths[h, :]
    exp_p   = m["Expected Price"]
    ci_lo   = m[f"CI Lower ({round(CONFIDENCE_LEVEL*100)}%)"]
    ci_hi   = m[f"CI Upper ({round(CONFIDENCE_LEVEL*100)}%)"]

    n_bins = min(100, int(N_SIMULATIONS ** 0.5))
    ax.hist(finals, bins=n_bins, density=True, color=BLUE, alpha=0.55, edgecolor="none")

    kde_x = np.linspace(finals.min(), finals.max(), 500)
    kde   = stats.gaussian_kde(finals)
    ax.plot(kde_x, kde(kde_x), color=GOLD, linewidth=2, label="KDE")

    ax.axvline(S0,     color=WHITE,  linewidth=1.4, linestyle=":",  label=f"S₀ {S0:,.0f}")
    ax.axvline(exp_p,  color=GREEN,  linewidth=1.4, linestyle="--", label=f"E[P] {exp_p:,.0f}")
    ax.axvline(ci_lo,  color=RED,    linewidth=1.0, linestyle="--", label=f"CI lo {ci_lo:,.0f}")
    ax.axvline(ci_hi,  color=ORANGE, linewidth=1.0, linestyle="--", label=f"CI hi {ci_hi:,.0f}")

    ax.set_title(f"{h}-day horizon", color=WHITE, fontsize=10)
    ax.set_xlabel("Price", color=WHITE)
    ax.set_ylabel("Density", color=WHITE)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    ax.legend(fontsize=7)
    ax.grid(True)

plt.tight_layout()
p4_path = os.path.join(OUTPUT_DIR, "04_final_dist.png")
plt.savefig(p4_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"📊  Plot 4 saved → {p4_path}\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 11 — Plot 5: VaR & CVaR Risk Dashboard            ║
# ╚══════════════════════════════════════════════════════════╝

fig, axes = plt.subplots(1, len(FORECAST_DAYS), figsize=(6 * len(FORECAST_DAYS), 5))
fig.suptitle(
    f"{TICKER}  —  VaR & CVaR Risk Dashboard  "
    f"({round(CONFIDENCE_LEVEL*100)}% confidence)",
    fontsize=13, color=GOLD, fontweight="bold")

if len(FORECAST_DAYS) == 1:
    axes = [axes]

for ax, h in zip(axes, FORECAST_DAYS):
    finals        = all_paths[h, :]
    returns_pct   = (finals - S0) / S0 * 100   # % returns

    var_threshold = float(np.percentile(returns_pct, (1 - CONFIDENCE_LEVEL) * 100))
    tail_vals     = returns_pct[returns_pct <= var_threshold]
    cvar_val      = float(np.mean(tail_vals)) if len(tail_vals) > 0 else var_threshold

    n_bins = min(100, int(N_SIMULATIONS ** 0.5))
    counts, bins, patches = ax.hist(returns_pct, bins=n_bins, density=True,
                                    color=BLUE, alpha=0.4, edgecolor="none")

    for patch, left in zip(patches, bins[:-1]):
        if left <= var_threshold:
            patch.set_facecolor(RED)
            patch.set_alpha(0.7)

    ax.axvline(0,             color=WHITE,  linewidth=1.2, linestyle=":")
    ax.axvline(var_threshold, color=ORANGE, linewidth=2.0, linestyle="--",
               label=f"VaR  = {var_threshold:.2f}%")
    ax.axvline(cvar_val,      color=RED,    linewidth=2.0, linestyle="--",
               label=f"CVaR = {cvar_val:.2f}%")

    kde_x = np.linspace(returns_pct.min(), returns_pct.max(), 500)
    kde   = stats.gaussian_kde(returns_pct)
    y_kde = kde(kde_x)
    mask  = kde_x <= var_threshold
    ax.fill_between(kde_x[mask], y_kde[mask], alpha=0.35, color=RED)
    ax.plot(kde_x, y_kde, color=GOLD, linewidth=1.5)

    ax.set_title(f"{h}-day horizon", color=WHITE, fontsize=10)
    ax.set_xlabel("Return (%)", color=WHITE)
    ax.set_ylabel("Density",    color=WHITE)
    ax.legend(fontsize=8)
    ax.grid(True)

plt.tight_layout()
p5_path = os.path.join(OUTPUT_DIR, "05_var_cvar.png")
plt.savefig(p5_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"📊  Plot 5 saved → {p5_path}\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 12 — Full Risk Summary Report (Console)           ║
# ╚══════════════════════════════════════════════════════════╝

def print_risk_report(ticker: str, S0: float, results: dict,
                      mu_annual: float, sig_annual: float):
    W = 62
    border = "═" * W

    print(f"\n{border}")
    print(f"  MONTE CARLO RISK REPORT  ·  {ticker}".center(W))
    print(f"  Generated : {datetime.today().strftime('%Y-%m-%d %H:%M')}".center(W))
    print(border)
    print(f"  Annual Expected Return  : {mu_annual * 100:>10.2f} %")
    print(f"  Annual Volatility (σ)   : {sig_annual * 100:>10.2f} %")
    print(f"  Sharpe Proxy (μ/σ ann.) : {mu_annual/sig_annual:>10.4f}")
    print(f"  Simulations Run         : {N_SIMULATIONS:>10,}")
    print(border)

    for h, m in results.items():
        conf_label = round(CONFIDENCE_LEVEL * 100)
        delta_exp  = m["Expected Price"] - S0
        delta_pct  = delta_exp / S0 * 100

        print(f"\n  ┌─── Horizon : {h} Trading Days {'─'*(W-28)}┐")
        print(f"  │  Starting Price             : {S0:>12,.4f}            │")
        print(f"  │  Expected Price (mean)      : {m['Expected Price']:>12,.4f}            │")
        print(f"  │  Median  Price              : {m['Median Price']:>12,.4f}            │")
        print(f"  │  Expected Δ                 : {delta_exp:>+12,.4f}  ({delta_pct:+.2f}%)     │")
        print(f"  │  Best Case  (95th %ile)     : {m['Best Case (95th %)']:>12,.4f}            │")
        print(f"  │  Worst Case ( 5th %ile)     : {m['Worst Case (5th %)']:>12,.4f}            │")
        print(f"  │  {conf_label}% CI                    : "
              f"[{m[f'CI Lower ({conf_label}%)']:>9,.2f}  "
              f"— {m[f'CI Upper ({conf_label}%)']:>9,.2f}]      │")
        print(f"  │  Probability of Profit      : {m['Prob. of Profit (%)']:>11.2f} %           │")
        print(f"  │  Probability of Loss        : {m['Prob. of Loss (%)']:>11.2f} %           │")
        print(f"  │  VaR  {conf_label}% (worst % loss)   : {m[f'VaR ({conf_label}%)']:>11.4f} %           │")
        print(f"  │  CVaR {conf_label}% (exp. tail loss) : {m[f'CVaR ({conf_label}%)']:>11.4f} %           │")
        print(f"  └{'─'*(W-2)}┘")

    print(f"\n{border}")
    print("  ⚠️  DISCLAIMER".center(W))
    print("  This is a statistical model for educational use.".center(W))
    print("  It is NOT financial advice.".center(W))
    print(f"{border}\n")


print_risk_report(TICKER, S0, results, mu_annual, sig_annual)


# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 13 — Export Results to CSV                        ║
# ╚══════════════════════════════════════════════════════════╝

export_rows = []
for h, m in results.items():
    row = {"Ticker": TICKER, **m}
    export_rows.append(row)

export_df = pd.DataFrame(export_rows)
csv_path  = os.path.join(OUTPUT_DIR, f"{TICKER}_monte_carlo_results.csv")
export_df.to_csv(csv_path, index=False)
print(f"💾  Results exported → {csv_path}")

for h in FORECAST_DAYS:
    finals     = all_paths[h, :]
    finals_df  = pd.DataFrame(finals, columns=[f"FinalPrice_{h}d"])
    fpath      = os.path.join(OUTPUT_DIR, f"{TICKER}_sim_finals_{h}d.csv")
    finals_df.to_csv(fpath, index=False)
    print(f"💾  Simulated finals ({h}d) → {fpath}")

print("\n✅  All exports complete.\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  BONUS — Interactive Plotly Fan Chart                   ║
# ╚══════════════════════════════════════════════════════════╝

try:
    import plotly.graph_objects as go

    PLOT_HORIZON = FORECAST_DAYS[-1]   # use the longest horizon
    days   = np.arange(PLOT_HORIZON + 1)
    paths  = all_paths[:PLOT_HORIZON + 1]

    p5  = np.percentile(paths, 5,  axis=1)
    p25 = np.percentile(paths, 25, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p75 = np.percentile(paths, 75, axis=1)
    p95 = np.percentile(paths, 95, axis=1)

    fig_pl = go.Figure()

    fig_pl.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself", fillcolor="rgba(76,163,255,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5th–95th %ile", hoverinfo="skip"))

    fig_pl.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself", fillcolor="rgba(76,163,255,0.22)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25th–75th %ile", hoverinfo="skip"))

    for i in range(min(50, N_SIMULATIONS)):
        fig_pl.add_trace(go.Scatter(
            x=days, y=paths[:, i],
            mode="lines",
            line=dict(color="rgba(76,163,255,0.04)", width=0.8),
            showlegend=False, hoverinfo="skip"))

    fig_pl.add_trace(go.Scatter(x=days, y=p50, mode="lines",
        line=dict(color="#FFD700", width=2), name="Median"))
    fig_pl.add_trace(go.Scatter(x=days, y=p95, mode="lines",
        line=dict(color="#00FF88", width=1.2, dash="dash"), name="95th %ile"))
    fig_pl.add_trace(go.Scatter(x=days, y=p5, mode="lines",
        line=dict(color="#FF4C4C", width=1.2, dash="dash"), name="5th %ile"))
    fig_pl.add_hline(y=S0, line_dash="dot", line_color="white",
                     annotation_text=f"S₀ = {S0:,.2f}",
                     annotation_font_color="white")

    fig_pl.update_layout(
        title=dict(
            text=f"<b>{TICKER} — Monte Carlo Simulation "
                 f"({N_SIMULATIONS:,} paths, {PLOT_HORIZON}d)</b>",
            font=dict(color="#FFD700", size=15)),
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#141414",
        font=dict(color="#cccccc", family="monospace"),
        xaxis=dict(title="Trading Days", gridcolor="#222"),
        yaxis=dict(title="Simulated Price", gridcolor="#222", tickformat=",.0f"),
        legend=dict(bgcolor="#1a1a1a", bordercolor="#444"),
        hovermode="x unified",
    )

    html_path = os.path.join(OUTPUT_DIR, f"{TICKER}_interactive_fan.html")
    fig_pl.write_html(html_path)
    print(f"🌐  Interactive chart → {html_path}\n")

except ImportError:
    print("Plotly not available — skipping interactive chart.\n")


# ╔══════════════════════════════════════════════════════════╗
# ║  BONUS — Portfolio Simulation (Multiple Tickers)        ║
# ╚══════════════════════════════════════════════════════════╝

PORTFOLIO_TICKERS = ["AAPL", "MSFT", "GOOGL"]
PORT_WEIGHTS      = [1/3, 1/3, 1/3]
PORT_HORIZON      = 252

print("═"*58)
print("  PORTFOLIO SIMULATION")
print("═"*58)

port_data = {}
for t in PORTFOLIO_TICKERS:
    try:
        raw = yf.download(t, period="5y", auto_adjust=True, progress=False)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" in raw.columns.levels[0]:
                    s = raw["Close"].squeeze()
                else:
                    raw.columns = raw.columns.get_level_values(0)
                    s = raw["Close"].squeeze()
            else:
                s = raw["Close"].squeeze()
            port_data[t] = np.log(s / s.shift(1)).dropna()
            print(f"  ✓  {t}")
        else:
            print(f"  ✗  {t}: No data returned")
    except Exception as e:
        print(f"  ✗  {t}: {e}")

if len(port_data) >= 2:
    ret_df    = pd.DataFrame(port_data).dropna()
    loaded    = list(ret_df.columns)
    w         = np.array([PORT_WEIGHTS[PORTFOLIO_TICKERS.index(t)] for t in loaded])
    w         /= w.sum()

    cov_daily = ret_df.cov().values
    mu_port   = ret_df.mean().values @ w
    sig_port  = np.sqrt(w @ cov_daily @ w)

    # Correlation matrix heatmap
    corr = ret_df.corr()
    n    = len(corr)
    fig, ax = plt.subplots(figsize=(max(5, n*1.5), max(4, n*1.3)))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.columns)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i,j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if abs(corr.values[i,j]) < 0.7 else "white")
    ax.set_title("Portfolio Correlation Matrix", color=GOLD, fontsize=12)
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#141414")
    plt.tight_layout()
    corr_path = os.path.join(OUTPUT_DIR, "06_correlation.png")
    plt.savefig(corr_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"\n📊  Correlation matrix saved → {corr_path}")

    # Run portfolio Monte Carlo
    port_paths = run_monte_carlo(100.0, mu_port + 0.5 * sig_port ** 2, sig_port,
                                 PORT_HORIZON, N_SIMULATIONS)
    port_final = port_paths[PORT_HORIZON]

    port_VaR  = float(np.percentile(
        (port_final - 100) / 100, (1 - CONFIDENCE_LEVEL) * 100) * 100)
    tail_port = port_final[port_final <= np.percentile(port_final, (1 - CONFIDENCE_LEVEL) * 100)]
    port_CVaR = float(np.mean((tail_port - 100) / 100) * 100) if len(tail_port) > 0 else port_VaR

    sharpe = (mu_port * 252) / (sig_port * np.sqrt(252))

    print(f"\n  Portfolio   : {', '.join(loaded)}")
    print(f"  Weights     : {[f'{x*100:.1f}%' for x in w]}")
    print(f"  Annual μ    : {mu_port*252*100:.2f}%")
    print(f"  Annual σ    : {sig_port*np.sqrt(252)*100:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.4f}")
    print(f"  VaR  {round(CONFIDENCE_LEVEL*100)}%    : {port_VaR:.2f}%")
    print(f"  CVaR {round(CONFIDENCE_LEVEL*100)}%    : {port_CVaR:.2f}%")
else:
    print("  Not enough valid tickers for portfolio analysis.")

print("\n" + "═"*58)
print("  ✅  PROJECT COMPLETE")
print("═"*58)
print(f"\n  All charts saved in {OUTPUT_DIR}/")
print(f"  CSV results : {os.path.join(OUTPUT_DIR, f'{TICKER}_monte_carlo_results.csv')}")
print(f"\n  ⚠️  Educational use only. Not financial advice.\n")
