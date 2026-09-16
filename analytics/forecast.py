"""Forecast Engine — explainable, driver-based (deterministic)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import calculation_engine as calc

def _monthly(df):
    ts = calc.timeseries(df, "M"); return ts.dropna(subset=["sales"]) if not ts.empty else ts

def forecast_next_periods(df, horizon=6, scenario="Base"):
    ts = _monthly(df)
    if ts.empty or len(ts) < 3: return {"ok": False, "reason": "Not enough history to forecast (need ≥3 months)."}
    ts = ts.sort_values("period").reset_index(drop=True); last = pd.Timestamp(ts["period"].iloc[-1]); recent = ts["sales"].tail(3).mean()
    mom = ts["sales"].pct_change().tail(3).mean(); run = 0.0 if np.isnan(mom) else float(mom)
    py_g = 0.0
    if "prior_year" in ts and ts["prior_year"].notna().any():
        py = ts.dropna(subset=["prior_year"])
        if len(py) >= 2 and py["prior_year"].iloc[-1]: py_g = float((py["sales"].iloc[-1] / py["prior_year"].iloc[-1]) - 1) / 12.0
    market = float(df["market_growth"].mean()) / 12.0 if ("market_growth" in df.columns and df["market_growth"].notna().any()) else 0.0
    budget = 0.0
    if "budget" in ts and ts["budget"].notna().any():
        b = ts.dropna(subset=["budget"])
        if len(b) and b["sales"].tail(3).mean(): budget = float((b["budget"].mean() / b["sales"].tail(3).mean()) - 1) / 12.0
    season = np.ones(12)
    if ts["sales"].notna().sum() >= 12:
        bm = ts.assign(m=pd.to_datetime(ts["period"]).dt.month).groupby("m")["sales"].mean()
        if bm.mean(): season = (bm / bm.mean()).reindex(range(1, 13)).fillna(1.0).values
    sens = {"Base": 1.0, "Upside": 1.5, "Downside": 0.5}.get(scenario, 1.0)
    drivers = {"Current run-rate": run * sens, "Prior-year trend": py_g * sens, "Market momentum": market * sens, "Budget anchor": budget}
    g = sum(drivers.values()); periods, values, base = [], [], recent
    for i in range(1, horizon + 1):
        p = last + pd.offsets.MonthBegin(i); base *= (1 + g); values.append(float(base * season[(p.month - 1) % 12] / np.mean(season))); periods.append(p)
    exp = pd.DataFrame({"Driver": list(drivers.keys()), "Monthly contribution": [f"{v*100:+.1f}%" for v in drivers.values()]})
    return {"ok": True, "scenario": scenario, "periods": periods, "values": values, "blended_monthly_growth": g, "drivers": drivers, "explanation": exp, "history": ts[["period", "sales"]], "forecast_total": float(np.sum(values))}

def scenario_bundle(df, horizon=6):
    return {s: forecast_next_periods(df, horizon, s) for s in ["Base", "Upside", "Downside"]}
