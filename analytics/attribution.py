"""Advanced Driver Attribution — fully-reconciling, with drill-through."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import calculation_engine as calc

TOL = 1.0


def _lines(df, actual=True):
    vcol = "volume" if actual else "budget_volume"
    rcol = "revenue" if actual else "budget_revenue"
    keys = [k for k in ["customer", "product"] if k in df.columns]
    if not keys or vcol not in df.columns or rcol not in df.columns:
        return None, keys
    return df.groupby(keys).agg(vol=(vcol, "sum"), rev=(rcol, "sum")).reset_index(), keys


def attribute_sales(df, base="budget"):
    actual = calc.sales(df)
    bc = "budget_revenue" if base == "budget" else "py_revenue"
    base_sales = calc._sum(df, bc)
    if np.isnan(base_sales):
        return {"ok": False, "reason": "No base to attribute against."}
    ga, keys = _lines(df, True)
    gb, _ = _lines(df, False) if base == "budget" else (None, keys)
    if base != "budget" and ga is not None and "py_volume" in df.columns:
        gb = df.groupby(keys).agg(vol=("py_volume", "sum"), rev=("py_revenue", "sum")).reset_index()
    if ga is None or gb is None:
        net = actual - base_sales
        return {"ok": True, "base": base_sales, "current": actual, "base_label": "Budget" if base == "budget" else "Prior year",
                "effects": {"Volume": net, "Price": 0.0, "Mix": 0.0, "New": 0.0, "Lost": 0.0, "Other / Residual": 0.0},
                "lines": pd.DataFrame(), "reconciles": True, "grain": keys}
    ga, gb = ga.copy(), gb.copy()
    ga["price_a"] = np.where(ga["vol"] != 0, ga["rev"] / ga["vol"], np.nan)
    gb["price_b"] = np.where(gb["vol"] != 0, gb["rev"] / gb["vol"], np.nan)
    merged = ga.merge(gb, on=keys, how="outer", suffixes=("_a", "_b"))
    a_p, b_p = merged["rev_a"].notna(), merged["rev_b"].notna(); both = a_p & b_p
    bop = base_sales / gb["vol"].sum() if gb["vol"].sum() else 0.0
    m = merged.fillna(0)
    vol = np.where(both, (m["vol_a"] - m["vol_b"]) * m["price_b"], 0.0)
    price = np.where(both, (m["price_a"] - m["price_b"]) * m["vol_a"], 0.0)
    mix = np.where(both, (m["price_b"] - bop) * (m["vol_a"] - m["vol_b"]), 0.0)
    vol = vol - mix
    new = np.where(a_p & ~b_p, m["rev_a"], 0.0); lost = np.where(~a_p & b_p, -m["rev_b"], 0.0)
    lt = m[keys].copy()
    lt["Volume"], lt["Price"], lt["Mix"], lt["New"], lt["Lost"] = vol, price, mix, new, lost
    lt["Actual"], lt["Base"] = m["rev_a"], m["rev_b"]; lt["Variance"] = m["rev_a"] - m["rev_b"]
    totals = {e: float(lt[e].sum()) for e in ["Volume", "Price", "Mix", "New", "Lost"]}
    totals["Other / Residual"] = float((actual - base_sales) - sum(totals.values()))
    return {"ok": True, "base": base_sales, "current": actual, "base_label": "Budget" if base == "budget" else "Prior year",
            "effects": totals, "lines": lt, "reconciles": abs(base_sales + sum(totals.values()) - actual) < TOL, "grain": keys}


def drilldown(attr, effect, by="customer", top_n=8):
    lines = attr.get("lines")
    if lines is None or lines.empty or effect not in lines.columns or by not in lines.columns:
        return pd.DataFrame()
    return lines.groupby(by)[effect].sum().reset_index().sort_values(effect, key=abs, ascending=False).head(top_n)


def bridge_steps(attr):
    order = ["Volume", "Price", "Mix", "New", "Lost", "Other / Residual"]
    return {"base_label": attr["base_label"], "base": attr["base"],
            "steps": [(e, attr["effects"][e]) for e in order if e in attr["effects"]],
            "current": attr["current"], "reconciles": attr["reconciles"]}
