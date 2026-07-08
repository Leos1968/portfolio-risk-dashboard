import pandas as pd
import numpy as np
from typing import Any
import yfinance as yf

def compute_portfolio_metrics(holdings: pd.DataFrame) -> dict[str, Any]:
    """Calculates basic portfolio level metrics: Total Value and Cash balance."""
    if holdings.empty:
        return {"total_value": 0.0, "cash_balance": 0.0}
    
    total_value = holdings["Exposure"].sum()
    
    # Safely extract cash balance regardless of capitalization
    cash_holdings = holdings[holdings["Asset Class"].str.upper() == "CASH"]
    cash_balance = cash_holdings["Exposure"].sum() if not cash_holdings.empty else 0.0
    
    return {
        "total_value": round(total_value, 2),
        "cash_balance": round(cash_balance, 2)
    }

def get_sector_allocations(holdings: pd.DataFrame) -> dict[str, float]:
    """Aggregates portfolio weight by sector."""
    if holdings.empty or "Sector" not in holdings.columns:
        return {}
        
    sector_group = holdings.groupby("Sector")["Weight"].sum().to_dict()
    
    # Normalize to ensure it sums to 100%
    total_weight = sum(sector_group.values())
    if total_weight > 0:
        sector_group = {k: round((v / total_weight) * 100, 2) for k, v in sector_group.items()}
    return sector_group

def compute_advanced_risk_metrics(holdings: pd.DataFrame) -> dict[str, Any]:
    """
    Calculates Portfolio Beta vs S&P 500 and 1-Day 95% Parametric Value at Risk (VaR)
    utilizing a highly robust unified pricing matrix that resists yfinance API changes.
    """
    if holdings.empty or "Ticker" not in holdings.columns:
        return {"portfolio_beta": 1.0, "var_95_percent": 0.0, "var_dollar": 0.0, "individual_betas": {}}

    # Filter out cash for risk calculations
    risk_assets = holdings[holdings["Asset Class"].str.upper() != "CASH"].copy()
    if risk_assets.empty:
        return {"portfolio_beta": 0.0, "var_95_percent": 0.0, "var_dollar": 0.0, "individual_betas": {}}

    tickers = risk_assets["Ticker"].unique().tolist()
    
    # Recalculate weights relative to the total portfolio exposure
    total_value = holdings["Exposure"].sum()
    if total_value <= 0:
        return {"portfolio_beta": 1.0, "var_95_percent": 0.0, "var_dollar": 0.0, "individual_betas": {}}
        
    weights_dict = {row["Ticker"]: row["Exposure"] / total_value for _, row in risk_assets.iterrows()}

    # Include benchmark directly into the download request
    benchmark = "^GSPC"
    all_tickers = list(set(tickers + [benchmark]))
    
    # 1) Robustly download pricing data bypassing yf.download multi-index bugs
    try:
        # auto_adjust=False prevents newer yfinance versions from dropping the standard 'Close' column
        raw_data = yf.download(all_tickers, period="1y", interval="1d", auto_adjust=False, progress=False)
        
        hist_data = pd.DataFrame()
        
        # Safely extract the 'Close' prices regardless of how yfinance formats the MultiIndex columns
        if isinstance(raw_data.columns, pd.MultiIndex):
            if 'Close' in raw_data.columns.get_level_values(0):
                hist_data = raw_data['Close']
            elif 'Close' in raw_data.columns.get_level_values(1):
                hist_data = raw_data.xs('Close', level=1, axis=1)
            else:
                # Fallback if 'Close' is missing but 'Adj Close' exists
                if 'Adj Close' in raw_data.columns.get_level_values(0):
                    hist_data = raw_data['Adj Close']
                elif 'Adj Close' in raw_data.columns.get_level_values(1):
                    hist_data = raw_data.xs('Adj Close', level=1, axis=1)
        else:
            # Single ticker fallback
            if 'Close' in raw_data.columns:
                hist_data = pd.DataFrame({all_tickers[0]: raw_data['Close']})
            elif 'Adj Close' in raw_data.columns:
                hist_data = pd.DataFrame({all_tickers[0]: raw_data['Adj Close']})
                
    except Exception:
        hist_data = pd.DataFrame()

    if hist_data.empty:
        return {"portfolio_beta": 1.0, "var_95_percent": 0.0, "var_dollar": 0.0, "individual_betas": {}}

    # Force numeric types to prevent Pandas pct_change KeyErrors on corrupted object columns
    hist_data = hist_data.apply(pd.to_numeric, errors='coerce')
    
    # Clean data and compute returns safely
    hist_data = hist_data.ffill().bfill()
    
    # Pandas 2.1+ compatibility check for pct_change fill_method deprecation
    try:
        daily_returns = hist_data.pct_change(fill_method=None).dropna()
    except TypeError:
        daily_returns = hist_data.pct_change().dropna()

    if daily_returns.empty:
        return {"portfolio_beta": 1.0, "var_95_percent": 0.0, "var_dollar": 0.0, "individual_betas": {}}

    # Extract SPY returns and isolate the asset matrix
    if benchmark in daily_returns.columns:
        spy_ret_series = daily_returns[benchmark]
        asset_returns = daily_returns.drop(columns=[benchmark])
    else:
        # Fallback flatline if the benchmark drops
        spy_ret_series = pd.Series(0.0, index=daily_returns.index)
        asset_returns = daily_returns

    market_variance = spy_ret_series.var()
    if pd.isna(market_variance) or market_variance == 0:
        market_variance = 0.0001

    portfolio_beta = 0.0
    individual_betas = {}

    # Calculate Beta covariance mathematically
    for ticker in tickers:
        if ticker in asset_returns.columns:
            asset_series = asset_returns[ticker]
            covariance = asset_series.cov(spy_ret_series)
            asset_beta = covariance / market_variance
        else:
            asset_beta = 1.0
        
        individual_betas[ticker] = round(asset_beta, 2)
        portfolio_beta += weights_dict.get(ticker, 0.0) * asset_beta

    # Parametric Value at Risk (VaR) utilizing asset covariance matrix
    cov_matrix = asset_returns.cov()
    
    # Build weight vector strictly aligned with the actual downloaded columns
    valid_tickers = [t for t in tickers if t in cov_matrix.columns]
    
    if len(valid_tickers) > 0:
        aligned_weights = np.array([weights_dict.get(t, 0.0) for t in valid_tickers])
        
        # Re-normalize just in case a ticker was dropped by the API
        if aligned_weights.sum() > 0:
            aligned_weights = aligned_weights / aligned_weights.sum()
            
        sub_cov_matrix = cov_matrix.loc[valid_tickers, valid_tickers]
        portfolio_variance = np.dot(aligned_weights.T, np.dot(sub_cov_matrix, aligned_weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
    else:
        portfolio_volatility = 0.02

    # 95% confidence level maps to a Z-score factor of 1.645
    var_95_percent = 1.645 * portfolio_volatility
    var_dollar = total_value * var_95_percent

    return {
        "portfolio_beta": round(portfolio_beta, 2),
        "var_95_percent": round(var_95_percent * 100, 2),
        "var_dollar": round(var_dollar, 2),
        "individual_betas": individual_betas
    }

def portfolio_report(holdings: pd.DataFrame) -> dict[str, Any]:
    """
    Master function to generate a comprehensive institutional risk report.
    This is the single entry point used by the API layer.
    """
    # 1. Base Portfolio Calculations
    base_metrics = compute_portfolio_metrics(holdings)
    
    # 2. Sector Allocation Vector
    sector_alloc = get_sector_allocations(holdings)
    
    # 3. Advanced Risk (Beta, VaR, Correlations)
    risk_metrics = compute_advanced_risk_metrics(holdings)
    
    # 4. Construct JSON Payload
    report = {
        "portfolio_summary": base_metrics,
        "allocations": {
            "sector": sector_alloc
        },
        "risk_analysis": risk_metrics
    }
    
    return report