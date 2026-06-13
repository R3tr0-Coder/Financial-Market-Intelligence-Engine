"""
Market Intelligence Engine — Pipeline Runner
=============================================
Executes the full analytics pipeline end-to-end and produces:
  1. CSV exports of processed data & signals
  2. A rich terminal report
  3. A multi-panel matplotlib figure (saved as PNG)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
import os
warnings.filterwarnings("ignore")

from engine import (
    MarketDataGenerator,
    TechnicalIndicators,
    StatisticalSignalDetector,
    PortfolioOptimizer,
    PerformanceAttribution,
    FactorDecomposer,
)

OUTPUT_DIR = "/mnt/user-data/outputs/market_intelligence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
RISK_FREE = 0.045

# ══════════════════════════════════════════════════════════
# STEP 1 — Generate synthetic market data
# ══════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  MARKET INTELLIGENCE ENGINE  ·  NumPy & Pandas Pipeline")
print("═"*65)
print(f"\n[1/7] Generating synthetic OHLCV data for {TICKERS} ...")

gen  = MarketDataGenerator(seed=99)
data = gen.generate(TICKERS, n_days=756)
print(f"      ✓ {len(data)} assets × {756} trading days generated")

# ══════════════════════════════════════════════════════════
# STEP 2 — Technical indicators
# ══════════════════════════════════════════════════════════
print("\n[2/7] Computing technical indicators ...")
enriched = {}
for ticker, df in data.items():
    enriched[ticker] = TechnicalIndicators.compute_all(df)
    print(f"      ✓ {ticker}: RSI, MACD, Bollinger, ATR, VWAP, Stochastic, OBV, ROC, RealVol")

# Build a multi-index dataframe for cross-asset analysis
close_prices = pd.DataFrame({t: enriched[t]["Close"] for t in TICKERS})
log_returns  = pd.DataFrame({t: enriched[t]["log_return"] for t in TICKERS})

# ══════════════════════════════════════════════════════════
# STEP 3 — Statistical signals
# ══════════════════════════════════════════════════════════
print("\n[3/7] Detecting statistical signals ...")
detector = StatisticalSignalDetector()

# Z-score signals for each asset
zscore_signals = pd.DataFrame({
    t: detector.zscore_signal(close_prices[t], window=30)
    for t in TICKERS
})

# Hurst exponents
hurst = {}
for t in TICKERS:
    h = detector.hurst_exponent(close_prices[t])
    regime_type = "Mean-Reverting" if h < 0.45 else ("Trending" if h > 0.55 else "Random Walk")
    hurst[t] = {"H": round(h, 4), "Regime": regime_type}
    print(f"      ✓ {t:6s}  Hurst={h:.4f}  →  {regime_type}")

# Pairs trading: AAPL vs MSFT
spread, hedge_ratio = detector.pairs_spread(close_prices["AAPL"], close_prices["MSFT"])
spread_zscore = detector.zscore_signal(spread, window=20)
print(f"\n      ✓ Pairs spread (AAPL/MSFT) hedge ratio: {hedge_ratio:.4f}")

# Volatility regimes
vol_regimes = pd.DataFrame({
    t: detector.volatility_regime(enriched[t]["RealVol_21"])
    for t in TICKERS
})
print(f"      ✓ Volatility regime classification complete")

# Anomaly detection
anomaly_scores = detector.anomaly_score(log_returns)
top_anomalies  = anomaly_scores.nlargest(5)
print(f"      ✓ Mahalanobis anomaly detection: top anomaly date = {top_anomalies.index[0].date()}")

# ══════════════════════════════════════════════════════════
# STEP 4 — Portfolio optimisation
# ══════════════════════════════════════════════════════════
print("\n[4/7] Running portfolio optimisation ...")
optimizer = PortfolioOptimizer(log_returns, risk_free_rate=RISK_FREE)

max_sharpe_port = optimizer.max_sharpe()
min_var_port    = optimizer.min_variance()
risk_parity_port = optimizer.risk_parity()
efficient_frontier = optimizer.efficient_frontier(n_points=300)

for p in [max_sharpe_port, min_var_port, risk_parity_port]:
    print(f"      ✓ {p['label']:15s}  "
          f"Ret={p['annual_ret']*100:.1f}%  "
          f"Vol={p['annual_vol']*100:.1f}%  "
          f"Sharpe={p['sharpe']:.3f}")

# ══════════════════════════════════════════════════════════
# STEP 5 — Factor decomposition
# ══════════════════════════════════════════════════════════
print("\n[5/7] Running PCA factor decomposition ...")
decomposer = FactorDecomposer(n_factors=3)
factor_returns, loadings = decomposer.fit_transform(log_returns)
exp_var = decomposer.explained_variance()
for f, v in exp_var.items():
    print(f"      ✓ {f}: explains {v*100:.1f}% of cross-asset variance")

# ══════════════════════════════════════════════════════════
# STEP 6 — Performance attribution
# ══════════════════════════════════════════════════════════
print("\n[6/7] Computing performance attribution ...")

# Max-Sharpe portfolio returns
ms_weights  = np.array(list(max_sharpe_port["weights"].values()))
port_ret    = log_returns.dropna() @ ms_weights
bench_ret   = log_returns["AAPL"].dropna()   # AAPL as benchmark proxy

attr = PerformanceAttribution(port_ret, bench_ret, risk_free_rate=RISK_FREE)
report_df = attr.full_report()
print("\n  ┌── Performance Report (Max Sharpe Portfolio) ──────────────┐")
for _, row in report_df.iterrows():
    print(f"  │  {row['Metric']:25s}  {row['Value']:>12s}          │")
print("  └───────────────────────────────────────────────────────────┘")

# ══════════════════════════════════════════════════════════
# STEP 7 — Visualisation
# ══════════════════════════════════════════════════════════
print("\n[7/7] Generating visualisations ...")

DARK  = "#0D1117"
PANEL = "#161B22"
GRID  = "#21262D"
ACCENT= "#58A6FF"
GREEN = "#3FB950"
RED   = "#F85149"
GOLD  = "#E3B341"
PURPLE= "#BC8CFF"
TEXT  = "#E6EDF3"
MUTED = "#8B949E"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       GRID,
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
    "font.size":        9,
})

fig = plt.figure(figsize=(20, 24), facecolor=DARK)
fig.suptitle("MARKET INTELLIGENCE ENGINE  ·  ANALYTICS DASHBOARD",
             fontsize=16, fontweight="bold", color=TEXT, y=0.98,
             fontfamily="monospace")

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.35,
                       left=0.06, right=0.97, top=0.95, bottom=0.04)

# ── Panel 1: Normalised price series ──
ax1 = fig.add_subplot(gs[0, :2])
colors_p = [ACCENT, GREEN, GOLD, RED, PURPLE, "#FF9900", "#00D4FF"]
for i, t in enumerate(TICKERS):
    norm = close_prices[t] / close_prices[t].iloc[0] * 100
    ax1.plot(norm.index, norm.values, label=t, color=colors_p[i],
             linewidth=1.2, alpha=0.85)
ax1.set_title("NORMALISED PRICE PERFORMANCE  (Base = 100)", color=TEXT,
              fontweight="bold", pad=8)
ax1.legend(loc="upper left", framealpha=0.3, ncol=4, fontsize=8)
ax1.grid(True, alpha=0.4)
ax1.set_ylabel("Indexed Price")

# ── Panel 2: Correlation heatmap ──
ax2 = fig.add_subplot(gs[0, 2])
corr = log_returns.corr()
mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
corr_masked = corr.copy()
corr_masked[mask_upper] = np.nan

im = ax2.imshow(corr_masked.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax2.set_xticks(range(len(TICKERS))); ax2.set_xticklabels(TICKERS, rotation=45, fontsize=7)
ax2.set_yticks(range(len(TICKERS))); ax2.set_yticklabels(TICKERS, fontsize=7)
for i in range(len(TICKERS)):
    for j in range(len(TICKERS)):
        if not mask_upper[i, j]:
            val = corr_masked.values[i, j]
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6.5, color="white" if abs(val) > 0.5 else "black")
plt.colorbar(im, ax=ax2, shrink=0.8)
ax2.set_title("RETURNS CORRELATION MATRIX", color=TEXT, fontweight="bold", pad=8)

# ── Panel 3: RSI for lead asset ──
ax3 = fig.add_subplot(gs[1, :2])
ticker_rsi = "NVDA"
rsi_data = enriched[ticker_rsi][["Close", "RSI_14"]].iloc[-252:]  # last year
ax3b = ax3.twinx()
ax3.plot(rsi_data.index, rsi_data["Close"], color=colors_p[5], linewidth=1.0, alpha=0.4, label="Price")
ax3b.plot(rsi_data.index, rsi_data["RSI_14"], color=ACCENT, linewidth=1.2, label="RSI-14")
ax3b.axhline(70, color=RED,   linewidth=0.8, linestyle="--", alpha=0.8)
ax3b.axhline(30, color=GREEN, linewidth=0.8, linestyle="--", alpha=0.8)
ax3b.axhspan(70, 100, alpha=0.06, color=RED)
ax3b.axhspan(0, 30,  alpha=0.06, color=GREEN)
ax3b.set_ylim(0, 100)
ax3b.set_ylabel("RSI", color=ACCENT)
ax3.set_ylabel("Price ($)", color=colors_p[5])
ax3.set_title(f"RSI-14 SIGNAL  ·  {ticker_rsi}  (Last 252 Days)", color=TEXT, fontweight="bold", pad=8)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left", framealpha=0.3)
ax3.grid(True, alpha=0.3)

# ── Panel 4: Hurst exponent bar chart ──
ax4 = fig.add_subplot(gs[1, 2])
h_vals  = [hurst[t]["H"]      for t in TICKERS]
h_colors= [GREEN if h < 0.45 else (RED if h > 0.55 else GOLD) for h in h_vals]
bars = ax4.barh(TICKERS, h_vals, color=h_colors, edgecolor=DARK, height=0.6)
ax4.axvline(0.5, color=MUTED, linewidth=1.0, linestyle="--", alpha=0.7)
ax4.set_xlim(0.3, 0.75)
for bar, val in zip(bars, h_vals):
    ax4.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=8, color=TEXT)
ax4.set_title("HURST EXPONENT\n(Green=Mean-Rev, Red=Trending)", color=TEXT,
              fontweight="bold", pad=8, fontsize=8)
ax4.set_xlabel("H Exponent")
ax4.grid(True, axis="x", alpha=0.3)

# ── Panel 5: Pairs spread Z-score ──
ax5 = fig.add_subplot(gs[2, :2])
sz = spread_zscore.iloc[-252:]
ax5.fill_between(sz.index, sz.values, 0,
                 where=sz.values > 0, color=GREEN, alpha=0.25, label="Long A/Short B")
ax5.fill_between(sz.index, sz.values, 0,
                 where=sz.values < 0, color=RED, alpha=0.25, label="Short A/Long B")
ax5.plot(sz.index, sz.values, color=ACCENT, linewidth=0.9)
ax5.axhline( 2.0, color=RED,   linewidth=0.8, linestyle="--", label="+2σ Entry")
ax5.axhline(-2.0, color=GREEN, linewidth=0.8, linestyle="--", label="-2σ Entry")
ax5.axhline( 0.5, color=MUTED, linewidth=0.5, linestyle=":")
ax5.axhline(-0.5, color=MUTED, linewidth=0.5, linestyle=":")
ax5.set_title("PAIRS TRADING Z-SCORE  ·  AAPL / MSFT  (Last 252 Days)", color=TEXT, fontweight="bold", pad=8)
ax5.set_ylabel("Z-Score (σ)")
ax5.legend(fontsize=7.5, loc="upper right", framealpha=0.3)
ax5.grid(True, alpha=0.3)

# ── Panel 6: Efficient frontier ──
ax6 = fig.add_subplot(gs[2, 2])
mc = efficient_frontier
sc = ax6.scatter(mc["annual_vol"]*100, mc["annual_ret"]*100,
                 c=mc["sharpe"], cmap="plasma", s=12, alpha=0.5, zorder=2)
plt.colorbar(sc, ax=ax6, label="Sharpe", shrink=0.8)
for p, color, marker in [
    (max_sharpe_port, GOLD,  "*"),
    (min_var_port,    GREEN, "D"),
    (risk_parity_port, PURPLE, "^"),
]:
    ax6.scatter(p["annual_vol"]*100, p["annual_ret"]*100,
                color=color, s=150, zorder=5, marker=marker,
                edgecolors="white", linewidth=0.8,
                label=p["label"])
ax6.set_xlabel("Annual Volatility (%)")
ax6.set_ylabel("Annual Return (%)")
ax6.set_title("EFFICIENT FRONTIER\n(★ Max Sharpe  ◆ Min Var  ▲ Risk Parity)", color=TEXT,
              fontweight="bold", pad=8, fontsize=8)
ax6.legend(fontsize=7.5, framealpha=0.3)
ax6.grid(True, alpha=0.3)

# ── Panel 7: Anomaly scores ──
ax7 = fig.add_subplot(gs[3, :2])
anom_plot = anomaly_scores.iloc[-252:]
threshold = anomaly_scores.quantile(0.95)
ax7.plot(anom_plot.index, anom_plot.values, color=ACCENT, linewidth=0.9, alpha=0.8)
ax7.fill_between(anom_plot.index, anom_plot.values, threshold,
                 where=anom_plot.values > threshold, color=RED, alpha=0.4,
                 label="Anomaly Zone (>95th pct)")
ax7.axhline(threshold, color=RED, linewidth=1.0, linestyle="--", alpha=0.9)
ax7.set_title("MAHALANOBIS DISTANCE ANOMALY DETECTOR  (Last 252 Days)", color=TEXT,
              fontweight="bold", pad=8)
ax7.set_ylabel("Distance Score")
ax7.legend(fontsize=8, framealpha=0.3)
ax7.grid(True, alpha=0.3)

# ── Panel 8: PCA factor loadings ──
ax8 = fig.add_subplot(gs[3, 2])
x_pos = np.arange(len(TICKERS))
width = 0.25
factor_colors = [ACCENT, GOLD, GREEN]
for i in range(3):
    offset = (i - 1) * width
    ax8.bar(x_pos + offset, loadings[f"Factor_{i+1}"].values,
            width=width, label=f"Factor {i+1} ({exp_var[f'Factor_{i+1}']*100:.0f}%)",
            color=factor_colors[i], alpha=0.85, edgecolor=DARK)
ax8.set_xticks(x_pos); ax8.set_xticklabels(TICKERS, fontsize=8)
ax8.axhline(0, color=MUTED, linewidth=0.6)
ax8.set_title("PCA FACTOR LOADINGS\n(Barra-style Risk Model)", color=TEXT,
              fontweight="bold", pad=8, fontsize=8)
ax8.set_ylabel("Loading Coefficient")
ax8.legend(fontsize=7.5, framealpha=0.3)
ax8.grid(True, axis="y", alpha=0.3)

plt.savefig(f"{OUTPUT_DIR}/dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=DARK, edgecolor="none")
print(f"      ✓ Dashboard saved → dashboard.png")

# ══════════════════════════════════════════════════════════
# EXPORT — CSVs
# ══════════════════════════════════════════════════════════
close_prices.to_csv(f"{OUTPUT_DIR}/prices.csv")
log_returns.to_csv(f"{OUTPUT_DIR}/log_returns.csv")
zscore_signals.to_csv(f"{OUTPUT_DIR}/zscore_signals.csv")
enriched["NVDA"].to_csv(f"{OUTPUT_DIR}/NVDA_enriched.csv")
report_df.to_csv(f"{OUTPUT_DIR}/performance_report.csv", index=False)
factor_returns.to_csv(f"{OUTPUT_DIR}/factor_returns.csv")
loadings.to_csv(f"{OUTPUT_DIR}/factor_loadings.csv")
anomaly_scores.to_frame().to_csv(f"{OUTPUT_DIR}/anomaly_scores.csv")

print(f"\n      ✓ CSVs exported to {OUTPUT_DIR}/")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  PIPELINE COMPLETE")
print("═"*65)
print(f"  Assets analysed:      {len(TICKERS)}")
print(f"  Trading days:         756  (~3 years)")
print(f"  Indicators/asset:     11")
print(f"  Total features:       {len(TICKERS) * 11:,}")
print(f"  Pairs spread:         AAPL / MSFT  (hr={hedge_ratio:.4f})")
print(f"  Top anomaly date:     {top_anomalies.index[0].date()}")
print(f"  Max Sharpe portfolio: {max_sharpe_port['sharpe']:.3f} Sharpe")
print(f"  Risk Parity Sharpe:   {risk_parity_port['sharpe']:.3f}")
print(f"  PCA Factor 1 var:     {exp_var['Factor_1']*100:.1f}%")
print("═"*65)
print()
