"""Predictive Budget Builder — deterministic, bottom-up."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import calculation_engine as calc

def _last12(df):
    ts = calc.timeseries(df, "M").dropna(subset=["sales"]).sort_values("period"); return ts.tail(12)

def build_budget(df, growth=0.06, target_gm=None, cost_inflation=0.03):
    ts = _last12(df)
    if ts.empty: return {"ok": False, "reason": "Need dated sales to build a budget."}
    lyr = float(ts["sales"].sum()); base_gm = target_gm if target_gm is not None else (calc.gm_pct(df) if not np.isnan(calc.gm_pct(df)) else 0.35)
    ny = int(pd.Timestamp(ts["period"].iloc[-1]).year) + 1; brev = lyr * (1 + growth)
    lastq = ts.assign(q=pd.to_datetime(ts["period"]).dt.quarter).groupby("q")["sales"].sum()
    w = (lastq / lastq.sum()) if lastq.sum() else pd.Series([0.25] * 4, index=[1, 2, 3, 4])
    qb = {f"{ny} Q{q}": float(brev * w.get(q, 0.25)) for q in [1, 2, 3, 4]}
    def split(dim):
        if dim not in df.columns: return pd.DataFrame()
        sh = df.groupby(dim)["revenue"].sum(); sh = sh / sh.sum() if sh.sum() else sh
        return pd.DataFrame({dim.title(): sh.index, "Share": sh.values, "Budget revenue": brev * sh.values, "Budget GM£": brev * sh.values * base_gm}).sort_values("Budget revenue", ascending=False)
    return {"ok": True, "next_year": ny, "last_year_rev": lyr, "growth": growth, "target_gm": base_gm, "cost_inflation": cost_inflation, "budget_rev": brev, "budget_gm_pounds": brev * base_gm, "budget_cogs": brev * (1 - base_gm) * (1 + cost_inflation), "q_budget": qb, "by_customer": split("customer"), "by_product": split("product"), "last_year_actual_quarters": {f"{int(pd.Timestamp(ts['period'].iloc[-1]).year)} Q{q}": float(lastq.get(q, 0)) for q in [1, 2, 3, 4]}}
