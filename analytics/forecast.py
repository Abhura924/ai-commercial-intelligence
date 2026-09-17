"""
Forecast Engine — explainable, driver-based (deterministic).

v0.4 addition: `forecast_frame()` projects the ACTUAL line-level data forward over
a horizon (months), scaling each (customer, product) row by its blended growth and
prior-year seasonality. This yields a canonical-shaped dataframe for the forecast
period, so the SAME scenario engine can run over a forecast horizon rather than only
the trailing actuals.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import calculation_engine as calc


def _monthly(df):
    ts = calc.timeseries(df, "M")
    return ts.dropna(subset=["sales"]) if not ts.empty else ts


def blended_growth(df, scenario="Base"):
    """Return (blended_monthly_growth, drivers dict) — the explainable growth build."""
    ts = _monthly(df)
    if ts.empty or len(ts) < 3:
        return 0.0, {}
    ts = ts.sort_values("period")
    mom = ts["sales"].pct_change().tail(3).mean()
    run = 0.0 if np.isnan(mom) else float(mom)
    py_g = 0.0
    if "prior_year" in ts and ts["prior_year"].notna().any():
        py = ts.dropna(subset=["prior_year"])
        if len(py) >= 2 and py["prior_year"].iloc[-1]:
            py_g = float((py["sales"].iloc[-1] / py["prior_year"].iloc[-1]) - 1) / 12.0
    market = float(df["market_growth"].mean()) / 12.0 if ("market_growth" in df.columns and df["market_growth"].notna().any()) else 0.0
    budget = 0.0
    if "budget" in ts and ts["budget"].notna().any():
        b = ts.dropna(subset=["budget"])
        if len(b) and b["sales"].tail(3).mean():
            budget = float((b["budget"].mean() / b["sales"].tail(3).mean()) - 1) / 12.0
    sens = {"Base": 1.0, "Upside": 1.5, "Downside": 0.5}.get(scenario, 1.0)
    drivers = {"Current run-rate": run * sens, "Prior-year trend": py_g * sens,
               "Market momentum": market * sens, "Budget anchor": budget}
    return sum(drivers.values()), drivers


def _seasonality(df):
    """Monthly seasonal factors (mean 1.0) from the actual data."""
    ts = _monthly(df)
    season = np.ones(12)
    if not ts.empty and ts["sales"].notna().sum() >= 12:
        bm = ts.assign(m=pd.to_datetime(ts["period"]).dt.month).groupby("m")["sales"].mean()
        if bm.mean():
            season = (bm / bm.mean()).reindex(range(1, 13)).fillna(1.0).values
    return season


def forecast_next_periods(df, horizon=6, scenario="Base"):
    ts = _monthly(df)
    if ts.empty or len(ts) < 3:
        return {"ok": False, "reason": "Not enough history to forecast (need ≥3 months)."}
    ts = ts.sort_values("period").reset_index(drop=True)
    last = pd.Timestamp(ts["period"].iloc[-1]); recent = ts["sales"].tail(3).mean()
    g, drivers = blended_growth(df, scenario)
    season = _seasonality(df)
    periods, values, base = [], [], recent
    for i in range(1, horizon + 1):
        p = last + pd.offsets.MonthBegin(i); base = base * (1 + g)
        values.append(float(base * season[(p.month - 1) % 12] / np.mean(season))); periods.append(p)
    exp = pd.DataFrame({"Driver": list(drivers.keys()),
                        "Monthly contribution": [f"{v*100:+.1f}%" for v in drivers.values()]})
    return {"ok": True, "scenario": scenario, "periods": periods, "values": values,
            "blended_monthly_growth": g, "drivers": drivers, "explanation": exp,
            "history": ts[["period", "sales"]], "forecast_total": float(np.sum(values))}


def scenario_bundle(df, horizon=6):
    return {s: forecast_next_periods(df, horizon, s) for s in ["Base", "Upside", "Downside"]}


# --------------------------------------------------------------------------- #
# v0.4 — build a canonical FORECAST FRAME the scenario engine can run over.
# --------------------------------------------------------------------------- #
def forecast_frame(df, horizon=6, scenario="Base"):
    """
    Roll the latest 12 months of actuals forward `horizon` months at row level.

    Each (customer, product) line keeps its structure (price, unit cost, GM),
    and its volume is grown by the blended monthly growth with prior-year
    seasonality. Budget/PY columns for the forecast period are set to the
    pre-scenario forecast so bridges/《vs budget》comparisons remain meaningful
    (variance vs the un-levered forecast). Returns a canonical dataframe.
    """
    if "date" not in df.columns:
        return {"ok": False, "reason": "No dates to build a forecast frame."}
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    last = pd.Timestamp(d["date"].max())
    window = d[d["date"] > last - pd.DateOffset(months=12)]
    if window.empty:
        window = d
    g, _ = blended_growth(df, scenario)
    season = _seasonality(df)
    season_mean = np.mean(season) if np.mean(season) else 1.0

    # Per (customer, product): average monthly volume, price, unit cost, gm rate
    keys = [k for k in ["customer", "product", "region", "business_unit"] if k in window.columns]
    if not keys:
        window = window.assign(_k="All"); keys = ["_k"]
    agg = window.groupby(keys).agg(
        volume=("volume", "mean"),
        price=("price", "mean") if "price" in window.columns else ("revenue", "mean"),
        revenue=("revenue", "mean"),
        cost=("cost", "mean"),
    ).reset_index()
    # derive price / unit cost robustly
    if "price" not in window.columns or agg["price"].isna().all():
        agg["price"] = np.where(agg["volume"] != 0, agg["revenue"] / agg["volume"], np.nan)
    agg["unit_cost"] = np.where(agg["volume"] != 0, agg["cost"] / agg["volume"], np.nan)
    opex_ratio = (calc.opex(window) / calc.sales(window)) if calc.sales(window) else 0.15
    if np.isnan(opex_ratio):
        opex_ratio = 0.15

    rows = []
    for i in range(1, horizon + 1):
        p = last + pd.offsets.MonthBegin(i)
        factor = (1 + g) ** i * season[(p.month - 1) % 12] / season_mean
        for _, r in agg.iterrows():
            vol = r["volume"] * factor
            price = r["price"]
            unit_cost = r["unit_cost"]
            rev = vol * price
            cost = vol * unit_cost
            row = {"date": p, "volume": vol, "price": price, "revenue": rev,
                   "cost": cost, "gross_margin": rev - cost, "opex": rev * opex_ratio,
                   # un-levered forecast acts as the 'budget' comparator for the period
                   "budget_volume": vol, "budget_revenue": rev, "budget_cost": cost,
                   "budget_opex": rev * opex_ratio,
                   "market_growth": float(df["market_growth"].mean()) if ("market_growth" in df.columns and df["market_growth"].notna().any()) else np.nan}
            for k in keys:
                row[k] = r[k]
            rows.append(row)
    fdf = pd.DataFrame(rows)
    if "_k" in fdf.columns:
        fdf = fdf.drop(columns=["_k"]); fdf["customer"] = "All"; fdf["product"] = "All"
    return {"ok": True, "frame": fdf.reset_index(drop=True), "horizon": horizon,
            "scenario": scenario, "blended_monthly_growth": g,
            "period_start": last + pd.offsets.MonthBegin(1),
            "period_end": last + pd.offsets.MonthBegin(horizon)}
