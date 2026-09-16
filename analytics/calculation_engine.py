"""Deterministic Calculation Engine — every number originates here."""
from __future__ import annotations
import numpy as np
import pandas as pd

def _sum(df, col): return float(df[col].sum()) if col in df.columns and df[col].notna().any() else np.nan
def sales(df): return _sum(df, "revenue")
def cogs(df): return _sum(df, "cost")
def gross_margin(df):
    if "gross_margin" in df.columns and df["gross_margin"].notna().any(): return _sum(df, "gross_margin")
    s, c = sales(df), cogs(df)
    return np.nan if (np.isnan(s) or np.isnan(c)) else s - c
def gm_pct(df):
    s, gm = sales(df), gross_margin(df)
    return np.nan if (np.isnan(s) or s == 0 or np.isnan(gm)) else gm / s
def opex(df): return _sum(df, "opex")
def ebitda(df):
    gm, ox = gross_margin(df), opex(df)
    return np.nan if np.isnan(gm) else gm - (0.0 if np.isnan(ox) else ox)
def volume(df): return _sum(df, "volume")
def avg_price(df):
    s, v = sales(df), volume(df)
    return np.nan if (np.isnan(s) or np.isnan(v) or v == 0) else s / v
def variance(a, b): return np.nan if (a is None or b is None or np.isnan(a) or np.isnan(b)) else a - b
def variance_pct(a, b): return np.nan if (a is None or b is None or np.isnan(a) or np.isnan(b) or b == 0) else (a - b) / b

def _budget(df, metric):
    if metric == "sales": return _sum(df, "budget_revenue")
    if metric == "gross_margin":
        if "budget_gross_margin" in df.columns and df["budget_gross_margin"].notna().any(): return _sum(df, "budget_gross_margin")
        br, bc = _sum(df, "budget_revenue"), _sum(df, "budget_cost")
        return br - bc if not (np.isnan(br) or np.isnan(bc)) else np.nan
    if metric == "gm_pct":
        bgm, brev = _budget(df, "gross_margin"), _sum(df, "budget_revenue")
        return bgm / brev if not (np.isnan(bgm) or np.isnan(brev) or brev == 0) else np.nan
    if metric == "ebitda":
        bgm, box = _budget(df, "gross_margin"), _sum(df, "budget_opex")
        return bgm - (0.0 if np.isnan(box) else box) if not np.isnan(bgm) else np.nan
    return np.nan

def _prior_year(df, metric):
    if metric == "sales": return _sum(df, "py_revenue")
    if metric == "gross_margin":
        if "py_gross_margin" in df.columns and df["py_gross_margin"].notna().any(): return _sum(df, "py_gross_margin")
        pr, pc = _sum(df, "py_revenue"), _sum(df, "py_cost")
        return pr - pc if not (np.isnan(pr) or np.isnan(pc)) else np.nan
    if metric == "gm_pct":
        pgm, prev = _prior_year(df, "gross_margin"), _sum(df, "py_revenue")
        return pgm / prev if not (np.isnan(pgm) or np.isnan(prev) or prev == 0) else np.nan
    if metric == "ebitda":
        pgm, pox = _prior_year(df, "gross_margin"), _sum(df, "py_opex")
        return pgm - (0.0 if np.isnan(pox) else pox) if not np.isnan(pgm) else np.nan
    return np.nan

def kpi_block(df):
    out = {}
    for key, val in {"sales": sales(df), "gross_margin": gross_margin(df), "gm_pct": gm_pct(df), "ebitda": ebitda(df)}.items():
        b, py = _budget(df, key), _prior_year(df, key)
        if key == "gm_pct":
            out[key] = {"value": val, "budget": b, "py": py, "variance": variance(val, b), "variance_pct": variance(val, b), "py_pct": variance(val, py), "is_ratio": True}
        else:
            out[key] = {"value": val, "budget": b, "py": py, "variance": variance(val, b), "variance_pct": variance_pct(val, b), "py_pct": variance_pct(val, py), "is_ratio": False}
    out["sales_vs_budget"] = {"value": out["sales"]["variance"], "budget": None, "py": None, "variance": out["sales"]["variance"], "variance_pct": out["sales"]["variance_pct"], "py_pct": None, "is_ratio": False}
    return out

def timeseries(df, freq="M"):
    if "date" not in df.columns: return pd.DataFrame()
    d = df.copy(); d["date"] = pd.to_datetime(d["date"])
    d["period"] = d["date"].dt.to_period("Q" if freq == "Q" else "M").dt.to_timestamp()
    g = d.groupby("period")
    ts = pd.DataFrame({"period": sorted(d["period"].unique())}).set_index("period")
    ts["sales"] = g["revenue"].sum() if "revenue" in d else np.nan
    ts["budget"] = g["budget_revenue"].sum() if "budget_revenue" in d else np.nan
    ts["prior_year"] = g["py_revenue"].sum() if "py_revenue" in d else np.nan
    if "revenue" in d:
        ts["gm_pct"] = g.apply(lambda x: gross_margin(x)) / g["revenue"].sum()
        if "budget_revenue" in d: ts["budget_gm_pct"] = g.apply(lambda x: _budget(x, "gm_pct"))
        if "py_revenue" in d: ts["py_gm_pct"] = g.apply(lambda x: _prior_year(x, "gm_pct"))
    return ts.reset_index()
