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
    """Calculates risk metrics using a flat, dictionary-based pricing fetcher."""
    tickers = holdings["Ticker"].unique().tolist()
    weights_dict = holdings.set_index("Ticker")["Weight"].to_dict()
    total_value = holdings["Exposure"].sum()
    
    if not tickers or total_value <= 0:
        return {"portfolio_beta": 1.0, "var_95_percent": 0.0, "var_dollar": 0.0, "individual_betas": {}}

    # 1. Fetch raw data as a flat dictionary to avoid MultiIndex/KeyError hell
    all_symbols = tickers + ["^GSPC"]
    raw_data = yf.download(all_symbols, period="1y", interval="1d", progress=False)["Close"]
    
    # 2. Flatten MultiIndex if yfinance decided to create one
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = [c[0] if isinstance(c, tuple) else c for c in raw_data.columns]
    
    # 3. Compute returns
    returns = raw_data.pct_change().dropna()
    
    # 4. Extract SPY and Assets
    spy_returns = returns["^GSPC"] if "^GSPC" in returns.columns else pd.Series(0, index=returns.index)
    asset_returns = returns.drop(columns=["^GSPC"], errors="ignore")
    
    # 5. Calculate Metrics
    portfolio_beta = 0.0
    individual_betas = {}
    
    for ticker in tickers:
        if ticker in asset_returns.columns:
            # Simple covariance calculation
            beta = asset_returns[ticker].cov(spy_returns) / spy_returns.var()
            individual_betas[ticker] = round(float(beta), 2)
            portfolio_beta += weights_dict.get(ticker, 0) * beta
        else:
            individual_betas[ticker] = 1.0
            portfolio_beta += weights_dict.get(ticker, 0) * 1.0

    # VaR calculation
    cov_matrix = asset_returns.cov()
    valid_tickers = [t for t in tickers if t in cov_matrix.index]
    weights_vec = np.array([weights_dict.get(t, 0) for t in valid_tickers])
    
    if len(valid_tickers) > 0:
        sub_cov = cov_matrix.loc[valid_tickers, valid_tickers]
        port_var = np.dot(weights_vec.T, np.dot(sub_cov, weights_vec))
        vol = np.sqrt(port_var)
    else:
        vol = 0.02
        
    return {
        "portfolio_beta": round(float(portfolio_beta), 2),
        "var_95_percent": round(float(1.645 * vol * 100), 2),
        "var_dollar": round(float(total_value * 1.645 * vol), 2),
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