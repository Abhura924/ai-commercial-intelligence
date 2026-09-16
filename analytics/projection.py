"""Long-Range (5-Year) Projection Engine — deterministic & explainable."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import calculation_engine as calc

def base_year(df):
    ts = calc.timeseries(df, "M").dropna(subset=["sales"])
    if ts.empty: return {"ok": False, "reason": "No dated sales to project from."}
    ts = ts.sort_values("period"); last12 = ts.tail(12); revenue = float(last12["sales"].sum())
    gm, st = calc.gross_margin(df), calc.sales(df); gmp = (gm / st) if st else np.nan; ox = calc.opex(df)
    return {"ok": True, "revenue": revenue, "gm_pct": float(gmp) if not np.isnan(gmp) else 0.35, "opex_ratio": float((ox / st) if (st and not np.isnan(ox)) else 0.15), "base_year": int(pd.Timestamp(last12["period"].iloc[-1]).year)}

def suggested_assumptions(df):
    ts = calc.timeseries(df, "M").dropna(subset=["sales"]); cagr = 0.08; py_g = np.nan
    if "prior_year" in ts and ts["prior_year"].notna().any():
        py = ts.dropna(subset=["prior_year"])
        if len(py) and py["prior_year"].sum(): py_g = float(py["sales"].sum() / py["prior_year"].sum() - 1)
    if not np.isnan(py_g): cagr = float(np.clip(py_g, -0.15, 0.35))
    elif len(ts) >= 6:
        mom = ts.sort_values("period")["sales"].pct_change().tail(6).mean()
        if not np.isnan(mom): cagr = float(np.clip((1 + mom) ** 12 - 1, -0.15, 0.35))
    market = float(df["market_growth"].mean()) if ("market_growth" in df.columns and df["market_growth"].notna().any()) else np.nan
    return {"revenue_cagr": round(cagr, 3), "gm_drift_pp": 0.0, "opex_efficiency_pp": 0.0, "market_growth": market}

def project(df, years=5, revenue_cagr=0.08, gm_drift_pp=0.0, opex_efficiency_pp=0.0):
    b = base_year(df)
    if not b.get("ok"): return {"ok": False, "reason": b.get("reason", "Cannot project.")}
    rows = []
    for n in range(1, years + 1):
        rev = b["revenue"] * (1 + revenue_cagr) ** n; gmp = float(np.clip(b["gm_pct"] + gm_drift_pp * n, 0.02, 0.95))
        gmpounds = rev * gmp; ox = float(max(0.0, b["opex_ratio"] - opex_efficiency_pp * n))
        rows.append({"year": b["base_year"] + n, "revenue": rev, "gm_pct": gmp, "gm_pounds": gmpounds, "opex_ratio": ox, "ebitda": gmpounds - rev * ox, "yoy": revenue_cagr})
    proj = pd.DataFrame(rows)
    return {"ok": True, "base": b, "years": years, "projection": proj, "cumulative_revenue": float(proj["revenue"].sum()), "cumulative_ebitda": float(proj["ebitda"].sum()), "exit_revenue": float(proj["revenue"].iloc[-1]), "exit_gm_pct": float(proj["gm_pct"].iloc[-1]), "exit_ebitda": float(proj["ebitda"].iloc[-1])}

def scenario_bundle(df, years=5, revenue_cagr=0.08, gm_drift_pp=0.0, opex_efficiency_pp=0.0, spread=0.03):
    return {name: project(df, years, revenue_cagr + cadj, gm_drift_pp + gadj, opex_efficiency_pp) for name, cadj, gadj in [("Base", 0.0, 0.0), ("Upside", spread, 0.002), ("Downside", -spread, -0.003)]}
