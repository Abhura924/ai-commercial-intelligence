"""
Driver-Level Scenario Engine (deterministic backend)
====================================================
Applies explicit member levers to the actual data and recomputes Sales, GM, GM%
and EBITDA through the SAME calculation engine, so every result stays traceable
and is never LLM-invented.

The AI Scenario Agent (ai/scenario_agent.py) turns natural language into the
structured `levers` this module executes — the LLM only extracts parameters, it
never does the arithmetic.

Lever schema (all keys optional except dimension+member, or scope='all'):
    dimension : "customer" | "product" | "region" | "business_unit"
    member    : specific name, OR "*"/None with scope='all' to apply company-wide
    volume_pct, price_pct, cost_pct : fractional changes (0.10 = +10%)
    churn     : bool — remove the member entirely
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import calculation_engine as calc


def blank_lever():
    return {"dimension": "customer", "member": None, "scope": "member",
            "volume_pct": 0.0, "price_pct": 0.0, "cost_pct": 0.0, "churn": False}


def _apply_row_levers(df, levers, opex_pct=0.0, opex_abs=0.0):
    d = df.copy()
    if "price" not in d.columns or d["price"].isna().all():
        d["price"] = np.where(d.get("volume", pd.Series(np.nan, index=d.index)) != 0, d["revenue"] / d["volume"], np.nan)
    if "volume" in d.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            d["_unit_cost"] = np.where(d["volume"] != 0, d["cost"] / d["volume"], np.nan)
    else:
        d["_unit_cost"] = np.nan

    keep = pd.Series(True, index=d.index)
    for lv in levers:
        dim = lv.get("dimension")
        scope = lv.get("scope", "member")
        member = lv.get("member")
        if scope == "all" or member in ("*", None, "all"):
            sel = pd.Series(True, index=d.index)  # company-wide
        else:
            if dim not in d.columns:
                continue
            sel = d[dim] == member
        if lv.get("churn"):
            keep &= ~sel
            continue
        if lv.get("volume_pct"):
            d.loc[sel, "volume"] = d.loc[sel, "volume"] * (1 + lv["volume_pct"])
        if lv.get("price_pct"):
            d.loc[sel, "price"] = d.loc[sel, "price"] * (1 + lv["price_pct"])
        if lv.get("cost_pct"):
            d.loc[sel, "_unit_cost"] = d.loc[sel, "_unit_cost"] * (1 + lv["cost_pct"])

    d = d[keep].copy()
    if "volume" in d.columns:
        d["revenue"] = d["volume"] * d["price"]
        d["cost"] = d["volume"] * d["_unit_cost"]
        d["gross_margin"] = d["revenue"] - d["cost"]
    if "opex" in d.columns:
        if opex_pct:
            d["opex"] = d["opex"] * (1 + opex_pct)
        if opex_abs:
            # spread an absolute OPEX change across rows in proportion to revenue
            total_rev = d["revenue"].sum()
            if total_rev:
                d["opex"] = d["opex"] + opex_abs * (d["revenue"] / total_rev)
            else:
                d["opex"] = d["opex"] + opex_abs / max(len(d), 1)
    return d.drop(columns=["_unit_cost"], errors="ignore")


def _metrics(df):
    return {"sales": calc.sales(df), "gross_margin": calc.gross_margin(df),
            "gm_pct": calc.gm_pct(df), "ebitda": calc.ebitda(df)}


def run_scenario(df, levers, opex_pct=0.0, opex_abs=0.0):
    baseline = _metrics(df)
    scen_df = _apply_row_levers(df, levers, opex_pct, opex_abs)
    scenario = _metrics(scen_df)
    deltas = {}
    for k in baseline:
        b, s = baseline[k], scenario[k]
        if k == "gm_pct":
            deltas[k] = {"delta": s - b, "unit": "pp"}
        else:
            deltas[k] = {"delta": s - b, "delta_pct": (s - b) / b if b else np.nan, "unit": "£"}
    return {"ok": True, "baseline": baseline, "scenario": scenario, "deltas": deltas,
            "scenario_df": scen_df, "_baseline_df": df, "levers": levers,
            "opex_pct": opex_pct, "opex_abs": opex_abs}


# --------------------------------------------------------------------------- #
# Scenario bridges (Baseline actuals -> Scenario) — all reconcile exactly.
# --------------------------------------------------------------------------- #
TOL = 1.0


def _line_grid(df):
    """(customer,product)-grain volume/revenue/cost for bridge building."""
    keys = [k for k in ["customer", "product"] if k in df.columns]
    if not keys:
        return None, keys
    g = df.groupby(keys).agg(vol=("volume", "sum"), rev=("revenue", "sum"),
                             cost=("cost", "sum")).reset_index()
    g["price"] = np.where(g["vol"] != 0, g["rev"] / g["vol"], np.nan)
    return g, keys


def scenario_sales_bridge(result):
    """Baseline sales -> scenario sales, decomposed Volume/Price/Mix/New/Lost/Other."""
    base_df, scen_df = result_baseline_df(result), result["scenario_df"]
    base_sales = calc.sales(base_df)
    scen_sales = calc.sales(scen_df)
    gb, keys = _line_grid(base_df)
    ga, _ = _line_grid(scen_df)
    if gb is None or ga is None:
        net = scen_sales - base_sales
        return {"base_label": "Baseline", "base": base_sales,
                "steps": [("Other / Residual", net)], "current": scen_sales, "reconciles": True}
    m = ga.merge(gb, on=keys, how="outer", suffixes=("_a", "_b"))
    a_present, b_present = m["rev_a"].notna(), m["rev_b"].notna()
    both = a_present & b_present
    m = m.fillna(0)
    bop = base_sales / gb["vol"].sum() if gb["vol"].sum() else 0.0
    vol = float(np.where(both, (m["vol_a"] - m["vol_b"]) * m["price_b"], 0).sum())
    price = float(np.where(both, (m["price_a"] - m["price_b"]) * m["vol_a"], 0).sum())
    mix = float(np.where(both, (m["price_b"] - bop) * (m["vol_a"] - m["vol_b"]), 0).sum())
    vol -= mix
    new = float(np.where(a_present & ~b_present, m["rev_a"], 0).sum())
    lost = float(np.where(~a_present & b_present, -m["rev_b"], 0).sum())
    steps = [("Volume", vol), ("Price", price), ("Mix", mix), ("New", new), ("Lost", lost)]
    steps.append(("Other / Residual", (scen_sales - base_sales) - sum(v for _, v in steps)))
    return {"base_label": "Baseline", "base": base_sales, "steps": steps,
            "current": scen_sales, "reconciles": abs(base_sales + sum(v for _, v in steps) - scen_sales) < TOL}


def scenario_gm_pct_bridge(result):
    """Baseline GM% -> scenario GM% in pp: price/volume&mix vs cost/raw-material."""
    base_df, scen_df = result_baseline_df(result), result["scenario_df"]
    gm_b, gm_a = calc.gm_pct(base_df), calc.gm_pct(scen_df)
    s_b, c_b = calc.sales(base_df), calc.cogs(base_df)
    s_a, c_a = calc.sales(scen_df), calc.cogs(scen_df)
    total = gm_a - gm_b
    steps = []
    if s_a and s_b:
        cost = -((c_a / s_a) - (c_b / s_b))
        steps = [("Price / volume & mix", total - cost), ("Cost / raw material", cost)]
    else:
        steps = [("Price / volume & mix", total)]
    steps.append(("Other / Residual", total - sum(v for _, v in steps)))
    return {"base_label": "Baseline", "base": gm_b, "steps": steps, "current": gm_a,
            "unit": "pp", "reconciles": abs(gm_b + sum(v for _, v in steps) - gm_a) < 1e-6}


def scenario_ebitda_bridge(result):
    """Baseline EBITDA -> scenario EBITDA: Sales/volume, Gross margin, OPEX, Other."""
    base_df, scen_df = result_baseline_df(result), result["scenario_df"]
    e_b, e_a = calc.ebitda(base_df), calc.ebitda(scen_df)
    s_b, s_a = calc.sales(base_df), calc.sales(scen_df)
    gmr_b, gmr_a = calc.gm_pct(base_df), calc.gm_pct(scen_df)
    ox_b, ox_a = calc.opex(base_df), calc.opex(scen_df)
    steps = []
    if not any(np.isnan(x) for x in [s_a, s_b, gmr_b]):
        steps.append(("Sales / volume impact", (s_a - s_b) * gmr_b))
    if not any(np.isnan(x) for x in [gmr_a, gmr_b, s_a]):
        steps.append(("Gross margin impact", (gmr_a - gmr_b) * s_a))
    if not (np.isnan(ox_a) or np.isnan(ox_b)):
        steps.append(("OPEX impact", -(ox_a - ox_b)))
    steps.append(("Other / Residual", (e_a - e_b) - sum(v for _, v in steps)))
    return {"base_label": "Baseline", "base": e_b, "steps": steps, "current": e_a,
            "reconciles": abs(e_b + sum(v for _, v in steps) - e_a) < TOL}


# Baseline df is captured on the result so bridges compare like-for-like.
def result_baseline_df(result):
    return result.get("_baseline_df")


def members(df, dimension):
    if dimension not in df.columns:
        return []
    return df.groupby(dimension)["revenue"].sum().sort_values(ascending=False).index.tolist()


def waterfall_vs_baseline(result):
    b, s = result["baseline"]["ebitda"], result["scenario"]["ebitda"]
    gm_delta = result["scenario"]["gross_margin"] - result["baseline"]["gross_margin"]
    steps = [("Gross margin Δ", gm_delta), ("OPEX Δ", (s - b) - gm_delta)]
    return {"base_label": "Baseline EBITDA", "base": b, "steps": steps, "current": s,
            "reconciles": abs(b + sum(v for _, v in steps) - s) < 1.0}
