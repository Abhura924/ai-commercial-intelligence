"""Bridge Engine — deterministic, reconciling bridges (Sales, GM%, EBITDA)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import calculation_engine as calc

TOL = 1.0


def _gpv(df, actual=True):
    vcol = "volume" if actual else "budget_volume"
    rcol = "revenue" if actual else "budget_revenue"
    keys = [k for k in ["customer", "product"] if k in df.columns]
    if not keys or vcol not in df.columns or rcol not in df.columns:
        return None
    g = df.groupby(keys).agg(vol=(vcol, "sum"), rev=(rcol, "sum")).reset_index()
    g["price"] = np.where(g["vol"] != 0, g["rev"] / g["vol"], np.nan)
    return g


def sales_bridge(df, base="budget"):
    actual = calc.sales(df)
    base_sales = calc._sum(df, "budget_revenue") if base == "budget" else calc._sum(df, "py_revenue")
    if np.isnan(base_sales):
        return {"base_label": base.title(), "base": actual, "steps": [("Other / Residual", 0.0)],
                "current": actual, "reconciles": True, "mode": "net"}
    steps = []
    ga = _gpv(df, True); gb = _gpv(df, False) if base == "budget" else None
    if ga is not None and gb is not None:
        keys = [k for k in ["customer", "product"] if k in df.columns]
        m = ga.merge(gb, on=keys, how="outer", suffixes=("_a", "_b")).fillna(0)
        vol = float(((m["vol_a"] - m["vol_b"]) * m["price_b"]).sum())
        price = float(((m["price_a"] - m["price_b"]) * m["vol_a"]).sum())
        bop = base_sales / m["vol_b"].sum() if m["vol_b"].sum() else 0
        mix = float(((m["price_b"] - bop) * (m["vol_a"] - m["vol_b"])).sum())
        vol -= mix
        steps += [("Volume", vol), ("Price", price), ("Mix", mix)]
    else:
        steps.append(("Volume", actual - base_sales))
    steps.append(("Other / Residual", (actual - base_sales) - sum(v for _, v in steps)))
    return {"base_label": "Budget" if base == "budget" else "Prior year", "base": base_sales,
            "steps": steps, "current": actual, "reconciles": abs(base_sales + sum(v for _, v in steps) - actual) < TOL, "mode": "full"}


def gm_pct_bridge(df, base="budget"):
    gm_a = calc.gm_pct(df)
    gm_b = calc._budget(df, "gm_pct") if base == "budget" else calc._prior_year(df, "gm_pct")
    if np.isnan(gm_a) or np.isnan(gm_b):
        return {"base_label": base.title(), "base": gm_b if not np.isnan(gm_b) else gm_a,
                "steps": [("Other / Residual", 0.0)], "current": gm_a, "reconciles": True, "unit": "pp"}
    total = gm_a - gm_b
    s_a, c_a = calc.sales(df), calc.cogs(df)
    s_b = calc._sum(df, "budget_revenue") if base == "budget" else calc._sum(df, "py_revenue")
    c_b = calc._sum(df, "budget_cost") if base == "budget" else calc._sum(df, "py_cost")
    steps = []
    if not any(np.isnan(x) for x in [s_a, c_a, s_b, c_b]) and s_a and s_b:
        cost = -((c_a / s_a) - (c_b / s_b))
        steps += [("Price / volume & mix", float(total - cost)), ("Cost / raw material", float(cost))]
    else:
        steps.append(("Price / volume & mix", float(total)))
    steps.append(("Other / Residual", float(total - sum(v for _, v in steps))))
    return {"base_label": "Budget" if base == "budget" else "Prior year", "base": gm_b, "steps": steps,
            "current": gm_a, "reconciles": abs(gm_b + sum(v for _, v in steps) - gm_a) < 1e-6, "unit": "pp"}


def ebitda_bridge(df, base="budget"):
    e_a = calc.ebitda(df)
    if base == "budget":
        e_b = calc._budget(df, "ebitda"); s_b = calc._sum(df, "budget_revenue")
        gmr_b = calc._budget(df, "gm_pct"); ox_b = calc._sum(df, "budget_opex")
    else:
        e_b = calc._prior_year(df, "ebitda"); s_b = calc._sum(df, "py_revenue")
        gmr_b = calc._prior_year(df, "gm_pct"); ox_b = calc._sum(df, "py_opex")
    if np.isnan(e_a) or np.isnan(e_b):
        return {"base_label": base.title(), "base": e_b if not np.isnan(e_b) else e_a,
                "steps": [("Other / Residual", 0.0)], "current": e_a, "reconciles": True}
    s_a, gmr_a, ox_a = calc.sales(df), calc.gm_pct(df), calc.opex(df)
    steps = []
    if not any(np.isnan(x) for x in [s_a, s_b, gmr_b]): steps.append(("Sales / volume impact", float((s_a - s_b) * gmr_b)))
    if not any(np.isnan(x) for x in [gmr_a, gmr_b, s_a]): steps.append(("Gross margin impact", float((gmr_a - gmr_b) * s_a)))
    if not (np.isnan(ox_a) or np.isnan(ox_b)): steps.append(("OPEX impact", float(-(ox_a - ox_b))))
    steps.append(("Other / Residual", float((e_a - e_b) - sum(v for _, v in steps))))
    return {"base_label": "Budget" if base == "budget" else "Prior year", "base": e_b, "steps": steps,
            "current": e_a, "reconciles": abs(e_b + sum(v for _, v in steps) - e_a) < TOL}


def driver_ranking(df, dimension="customer", base="budget", top_n=5):
    if dimension not in df.columns: return {"positive": [], "negative": []}
    bc = "budget_revenue" if base == "budget" else "py_revenue"
    if bc not in df.columns: return {"positive": [], "negative": []}
    g = df.groupby(dimension).agg(actual=("revenue", "sum"), base=(bc, "sum")).reset_index()
    g["variance"] = g["actual"] - g["base"]; g = g.sort_values("variance", ascending=False)
    pos = [(r[dimension], float(r["variance"])) for _, r in g[g["variance"] > 0].head(top_n).iterrows()]
    neg = [(r[dimension], float(r["variance"])) for _, r in g[g["variance"] < 0].tail(top_n).iloc[::-1].iterrows()]
    return {"positive": pos, "negative": neg}
