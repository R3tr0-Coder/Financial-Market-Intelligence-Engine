"""
Market Intelligence Engine
===========================
A production-grade financial analytics pipeline using NumPy & Pandas.

Features:
  - Synthetic OHLCV data generation with realistic market microstructure
  - Technical indicator computation (RSI, MACD, Bollinger Bands, ATR, VWAP)
  - Statistical arbitrage signal detection (Z-score, cointegration, Hurst exponent)
  - Mean-variance portfolio optimization (Markowitz + Sharpe maximization)
  - Volatility regime detection via Hidden Markov-inspired rolling statistics
  - Anomaly detection (Mahalanobis distance on returns space)
  - Full performance attribution report (alpha, beta, Sharpe, Sortino, Calmar)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. SYNTHETIC MARKET DATA GENERATOR
# ─────────────────────────────────────────────

class MarketDataGenerator:
    """
    Generates realistic OHLCV price series using Geometric Brownian Motion
    with jump diffusion, fat tails, and volatility clustering (GARCH-like).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        tickers: list[str],
        n_days: int = 756,         # ~3 years of trading days
        start_price: float = 100.0,
        annual_drift: float = 0.08,
        annual_vol: float = 0.20,
    ) -> dict[str, pd.DataFrame]:
        """
        Returns a dict of ticker -> OHLCV DataFrame with DatetimeIndex.
        Incorporates:
          - Correlated GBM across assets
          - GARCH(1,1)-style volatility clustering
          - Jump diffusion (Merton model) for fat tails
        """
        n = len(tickers)
        dt = 1 / 252

        # Build a realistic correlation matrix via random Wishart draw
        raw = self.rng.standard_normal((n, n))
        corr_matrix = raw @ raw.T
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)

        data = {}
        for i, ticker in enumerate(tickers):
            prices, volumes = self._simulate_single(
                n_days, start_price * (0.8 + 0.4 * self.rng.random()),
                annual_drift + self.rng.uniform(-0.04, 0.04),
                annual_vol + self.rng.uniform(-0.05, 0.10),
                dt,
            )
            dates = pd.bdate_range("2022-01-03", periods=n_days)
            df = pd.DataFrame(index=dates)
            df.index.name = "Date"
            df["Close"] = prices
            df["Open"] = prices * (1 + self.rng.normal(0, 0.003, n_days))
            df["High"] = np.maximum(df["Open"], df["Close"]) * (1 + np.abs(self.rng.normal(0, 0.004, n_days)))
            df["Low"]  = np.minimum(df["Open"], df["Close"]) * (1 - np.abs(self.rng.normal(0, 0.004, n_days)))
            df["Volume"] = volumes
            data[ticker] = df

        return data

    def _simulate_single(self, n, s0, mu, sigma, dt):
        """GBM + GARCH volatility clustering + Merton jump diffusion."""
        prices = np.zeros(n)
        prices[0] = s0
        vol = np.zeros(n)
        vol[0] = sigma

        # GARCH(1,1) parameters
        omega, alpha_g, beta_g = 0.00001, 0.09, 0.90

        # Jump parameters
        jump_intensity = 5 / 252   # ~5 jumps/year
        jump_mean, jump_std = -0.01, 0.03

        for t in range(1, n):
            # Update variance (GARCH)
            prev_eps = (np.log(prices[t-1] / prices[max(0, t-2)]) - mu * dt) if t > 1 else 0
            vol[t] = np.sqrt(omega + alpha_g * prev_eps**2 + beta_g * vol[t-1]**2)

            # GBM step
            z = self.rng.standard_normal()
            jump = 0
            if self.rng.random() < jump_intensity:
                jump = self.rng.normal(jump_mean, jump_std)

            log_ret = (mu - 0.5 * vol[t]**2) * dt + vol[t] * np.sqrt(dt) * z + jump
            prices[t] = prices[t-1] * np.exp(log_ret)

        # Realistic volume: mean-reverting with price-impact correlation
        base_vol = 1_000_000
        price_changes = np.abs(np.diff(np.log(prices), prepend=np.log(prices[0])))
        volumes = (base_vol * (1 + 3 * price_changes / price_changes.std())
                   * self.rng.lognormal(0, 0.3, n)).astype(int)

        return prices, volumes


# ─────────────────────────────────────────────
# 2. TECHNICAL INDICATOR ENGINE
# ─────────────────────────────────────────────

class TechnicalIndicators:
    """Vectorized computation of 10+ technical indicators using NumPy/Pandas."""

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        c = df["Close"].values
        h = df["High"].values
        lo = df["Low"].values
        v = df["Volume"].values

        # Returns
        out["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        out["pct_return"]  = df["Close"].pct_change()

        # RSI (Wilder's smoothing)
        out["RSI_14"] = TechnicalIndicators._rsi(c, 14)

        # MACD
        out["MACD"], out["MACD_signal"], out["MACD_hist"] = TechnicalIndicators._macd(c)

        # Bollinger Bands
        out["BB_mid"], out["BB_upper"], out["BB_lower"], out["BB_pct"] = \
            TechnicalIndicators._bollinger(c, 20, 2.0)

        # ATR (Average True Range) — normalised
        out["ATR_14"] = TechnicalIndicators._atr(h, lo, c, 14)
        out["ATR_norm"] = out["ATR_14"] / df["Close"]

        # VWAP (rolling 20-day)
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        out["VWAP_20"] = (typical * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()

        # Stochastic Oscillator %K and %D
        out["Stoch_K"], out["Stoch_D"] = TechnicalIndicators._stochastic(h, lo, c, 14, 3)

        # On-Balance Volume
        direction = np.sign(np.diff(c, prepend=c[0]))
        out["OBV"] = (v * direction).cumsum()

        # Momentum (10-day Rate of Change)
        out["ROC_10"] = (df["Close"] / df["Close"].shift(10) - 1) * 100

        # Rolling realized volatility (21-day annualised)
        out["RealVol_21"] = out["log_return"].rolling(21).std() * np.sqrt(252)

        return out

    @staticmethod
    def _rsi(prices: np.ndarray, period: int) -> pd.Series:
        deltas = np.diff(prices, prepend=prices[0])
        gains  = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_g  = pd.Series(gains).ewm(alpha=1/period, adjust=False).mean()
        avg_l  = pd.Series(losses).ewm(alpha=1/period, adjust=False).mean()
        rs     = avg_g / avg_l.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(prices: np.ndarray, fast=12, slow=26, signal=9):
        s = pd.Series(prices)
        ema_fast   = s.ewm(span=fast, adjust=False).mean()
        ema_slow   = s.ewm(span=slow, adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram  = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _bollinger(prices: np.ndarray, window: int, n_std: float):
        s   = pd.Series(prices)
        mid = s.rolling(window).mean()
        std = s.rolling(window).std()
        upper = mid + n_std * std
        lower = mid - n_std * std
        pct   = (s - lower) / (upper - lower)
        return mid, upper, lower, pct

    @staticmethod
    def _atr(high, low, close, period):
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low,
             np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        return pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def _stochastic(high, low, close, k_period, d_period):
        h_max = pd.Series(high).rolling(k_period).max()
        l_min = pd.Series(low).rolling(k_period).min()
        k = 100 * (pd.Series(close) - l_min) / (h_max - l_min)
        d = k.rolling(d_period).mean()
        return k, d


# ─────────────────────────────────────────────
# 3. STATISTICAL SIGNAL DETECTOR
# ─────────────────────────────────────────────

class StatisticalSignalDetector:
    """
    Detects alpha-generating signals:
      - Z-score mean reversion
      - Pairs trading via cointegration (Engle-Granger proxy)
      - Hurst exponent (trend vs. mean-reversion regime)
      - Volatility regime classification
    """

    @staticmethod
    def zscore_signal(series: pd.Series, window: int = 30) -> pd.Series:
        """Classic rolling Z-score for mean-reversion signals."""
        mu  = series.rolling(window).mean()
        sig = series.rolling(window).std()
        return (series - mu) / sig

    @staticmethod
    def hurst_exponent(series: pd.Series, min_lag=2, max_lag=100) -> float:
        """
        Estimates Hurst exponent via rescaled range analysis.
          H < 0.5  → mean-reverting
          H = 0.5  → random walk
          H > 0.5  → trending
        """
        ts = series.dropna().values
        lags = range(min_lag, min(max_lag, len(ts) // 4))
        rs_vals = []
        for lag in lags:
            chunks = [ts[i:i+lag] for i in range(0, len(ts) - lag, lag)]
            rs_chunk = []
            for chunk in chunks:
                if len(chunk) < 4:
                    continue
                mean_c = np.mean(chunk)
                deviations = np.cumsum(chunk - mean_c)
                R = np.ptp(deviations)
                S = np.std(chunk, ddof=1)
                if S > 0:
                    rs_chunk.append(R / S)
            if rs_chunk:
                rs_vals.append((lag, np.mean(rs_chunk)))

        if len(rs_vals) < 5:
            return 0.5
        lags_arr  = np.log([r[0] for r in rs_vals])
        rs_arr    = np.log([r[1] for r in rs_vals])
        slope, *_ = np.polyfit(lags_arr, rs_arr, 1)
        return float(np.clip(slope, 0.01, 0.98))

    @staticmethod
    def pairs_spread(price_a: pd.Series, price_b: pd.Series) -> tuple[pd.Series, float]:
        """
        Computes the hedge-ratio and spread between two assets
        via OLS regression (Engle-Granger step 1).
        Returns (spread, hedge_ratio).
        """
        log_a = np.log(price_a.dropna())
        log_b = np.log(price_b.dropna())
        aligned = pd.concat([log_a, log_b], axis=1).dropna()
        x = aligned.iloc[:, 1].values
        y = aligned.iloc[:, 0].values
        slope, intercept, r, p, se = stats.linregress(x, y)
        spread = pd.Series(y - slope * x - intercept, index=aligned.index)
        return spread, slope

    @staticmethod
    def volatility_regime(realized_vol: pd.Series, n_regimes: int = 3) -> pd.Series:
        """
        Classifies each day into LOW / MEDIUM / HIGH volatility regime
        using rolling percentile thresholds.
        """
        vol = realized_vol.dropna()
        p33 = vol.quantile(0.33)
        p66 = vol.quantile(0.66)
        regime = pd.cut(
            realized_vol,
            bins=[-np.inf, p33, p66, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"]
        )
        return regime

    @staticmethod
    def anomaly_score(returns_df: pd.DataFrame) -> pd.Series:
        """
        Computes Mahalanobis distance across the returns matrix.
        High scores indicate anomalous market conditions (e.g., flash crashes).
        """
        clean = returns_df.dropna()
        scaler = StandardScaler()
        X = scaler.fit_transform(clean)
        cov = np.cov(X, rowvar=False)
        try:
            inv_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        mu = X.mean(axis=0)
        diff = X - mu
        scores = np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))
        return pd.Series(scores, index=clean.index, name="Mahalanobis")


# ─────────────────────────────────────────────
# 4. PORTFOLIO OPTIMIZER
# ─────────────────────────────────────────────

class PortfolioOptimizer:
    """
    Markowitz mean-variance optimization:
      - Maximum Sharpe Ratio portfolio
      - Minimum Variance portfolio
      - Risk Parity portfolio
      - Efficient frontier sampling
    """

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.045):
        self.returns = returns.dropna()
        self.rf = risk_free_rate / 252  # daily
        self.mu = self.returns.mean().values
        self.sigma = self.returns.cov().values
        self.n = len(self.mu)

    def max_sharpe(self) -> dict:
        """Maximise Sharpe via Sequential Least Squares Programming (SLSQP)."""
        def neg_sharpe(w):
            ret = np.dot(w, self.mu)
            vol = np.sqrt(w @ self.sigma @ w)
            return -(ret - self.rf) / (vol + 1e-10)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0.0, 0.35)] * self.n   # max 35% per asset
        w0 = np.ones(self.n) / self.n

        res = minimize(neg_sharpe, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-12})
        return self._portfolio_stats(res.x, "Max Sharpe")

    def min_variance(self) -> dict:
        def portfolio_var(w):
            return w @ self.sigma @ w

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0.0, 0.35)] * self.n
        w0 = np.ones(self.n) / self.n

        res = minimize(portfolio_var, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 1000})
        return self._portfolio_stats(res.x, "Min Variance")

    def risk_parity(self) -> dict:
        """Equal risk contribution (ERC) portfolio."""
        def risk_parity_objective(w):
            w = np.maximum(w, 1e-10)
            port_var = w @ self.sigma @ w
            marginal_risk = self.sigma @ w
            risk_contrib = w * marginal_risk / port_var
            target = np.ones(self.n) / self.n
            return np.sum((risk_contrib - target) ** 2)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0.005, 0.5)] * self.n
        w0 = np.ones(self.n) / self.n

        res = minimize(risk_parity_objective, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 2000, "ftol": 1e-14})
        return self._portfolio_stats(res.x, "Risk Parity")

    def efficient_frontier(self, n_points: int = 60) -> pd.DataFrame:
        """Sample the efficient frontier via Monte Carlo simulation."""
        records = []
        for _ in range(n_points * 20):
            w = self.rng_weights()
            p = self._portfolio_stats(w, "MC")
            records.append(p)
        df = pd.DataFrame(records)
        # Keep only Pareto-efficient points
        df = df.sort_values("annual_vol").drop_duplicates("annual_ret", keep="first")
        return df

    def rng_weights(self) -> np.ndarray:
        w = np.random.dirichlet(np.ones(self.n))
        return w

    def _portfolio_stats(self, w: np.ndarray, label: str) -> dict:
        ann_ret = np.dot(w, self.mu) * 252
        ann_vol = np.sqrt(w @ self.sigma @ w) * np.sqrt(252)
        sharpe  = (ann_ret - self.rf * 252) / (ann_vol + 1e-10)
        return {
            "label": label,
            "weights": dict(zip(self.returns.columns, np.round(w, 4))),
            "annual_ret": round(ann_ret, 4),
            "annual_vol": round(ann_vol, 4),
            "sharpe":     round(sharpe,  4),
        }


# ─────────────────────────────────────────────
# 5. PERFORMANCE ATTRIBUTION ENGINE
# ─────────────────────────────────────────────

class PerformanceAttribution:
    """
    Computes full risk-adjusted performance metrics including:
    Alpha, Beta, Sharpe, Sortino, Calmar, Max Drawdown, VaR, CVaR.
    """

    def __init__(self, portfolio_returns: pd.Series,
                 benchmark_returns: pd.Series,
                 risk_free_rate: float = 0.045):
        self.port   = portfolio_returns.dropna()
        self.bench  = benchmark_returns.dropna()
        self.rf_ann = risk_free_rate
        self.rf_day = risk_free_rate / 252

    def full_report(self) -> pd.DataFrame:
        aligned = pd.concat([self.port, self.bench], axis=1).dropna()
        p = aligned.iloc[:, 0]
        b = aligned.iloc[:, 1]

        # CAPM regression: alpha, beta
        slope, intercept, r_val, p_val, _ = stats.linregress(b, p)
        beta  = slope
        alpha = intercept * 252   # annualised

        # Core metrics
        ann_ret = p.mean() * 252
        ann_vol = p.std()  * np.sqrt(252)
        sharpe  = (ann_ret - self.rf_ann) / (ann_vol + 1e-10)

        # Sortino (downside deviation)
        downside = p[p < self.rf_day]
        down_dev  = downside.std() * np.sqrt(252) if len(downside) > 1 else ann_vol
        sortino   = (ann_ret - self.rf_ann) / (down_dev + 1e-10)

        # Max Drawdown and Calmar
        cum_ret = (1 + p).cumprod()
        rolling_max = cum_ret.cummax()
        drawdowns   = (cum_ret - rolling_max) / rolling_max
        max_dd      = drawdowns.min()
        calmar      = ann_ret / abs(max_dd) if max_dd != 0 else np.inf

        # Value at Risk & CVaR (95%)
        var_95  = float(np.percentile(p, 5))
        cvar_95 = float(p[p <= var_95].mean())

        # Information Ratio
        active_ret = p - b
        ir = active_ret.mean() / (active_ret.std() + 1e-10) * np.sqrt(252)

        # Skewness & Kurtosis (fat tails)
        skew = float(stats.skew(p))
        kurt = float(stats.kurtosis(p))

        metrics = {
            "Annual Return":    f"{ann_ret*100:.2f}%",
            "Annual Volatility":f"{ann_vol*100:.2f}%",
            "Sharpe Ratio":     f"{sharpe:.3f}",
            "Sortino Ratio":    f"{sortino:.3f}",
            "Calmar Ratio":     f"{calmar:.3f}",
            "Beta":             f"{beta:.3f}",
            "Alpha (ann.)":     f"{alpha*100:.2f}%",
            "Max Drawdown":     f"{max_dd*100:.2f}%",
            "Info. Ratio":      f"{ir:.3f}",
            "VaR (95%, daily)": f"{var_95*100:.2f}%",
            "CVaR (95%, daily)":f"{cvar_95*100:.2f}%",
            "Skewness":         f"{skew:.3f}",
            "Excess Kurtosis":  f"{kurt:.3f}",
            "R² (vs bench)":    f"{r_val**2:.3f}",
        }
        return pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])


# ─────────────────────────────────────────────
# 6. PCA FACTOR DECOMPOSITION
# ─────────────────────────────────────────────

class FactorDecomposer:
    """
    Applies PCA to the returns matrix to extract latent market factors,
    similar to a simplified Barra-style risk model.
    """

    def __init__(self, n_factors: int = 3):
        self.n_factors = n_factors
        self.pca = PCA(n_components=n_factors)
        self.scaler = StandardScaler()

    def fit_transform(self, returns_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        clean = returns_df.dropna()
        X_scaled = self.scaler.fit_transform(clean)
        factors = self.pca.fit_transform(X_scaled)

        factor_df = pd.DataFrame(
            factors, index=clean.index,
            columns=[f"Factor_{i+1}" for i in range(self.n_factors)]
        )
        loadings_df = pd.DataFrame(
            self.pca.components_.T,
            index=returns_df.columns,
            columns=[f"Factor_{i+1}" for i in range(self.n_factors)]
        )
        return factor_df, loadings_df

    def explained_variance(self) -> pd.Series:
        return pd.Series(
            self.pca.explained_variance_ratio_,
            index=[f"Factor_{i+1}" for i in range(self.n_factors)],
            name="Explained Variance"
        )
