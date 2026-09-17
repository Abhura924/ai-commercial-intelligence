"""
Driver-Level Scenario Engine (deterministic backend).
Applies member levers to a base dataframe (actuals OR a forecast frame) and
recomputes Sales, GM, GM%, EBITDA through the SAME calculation engine.
Includes reconciling scenario bridges (baseline -> scenario).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import calculation_engine as calc

TOL = 1.0


def blank_lever():
    return {"dimension": "customer", "member": None, "scope": "member",
            "volume_pct": 0.0, "price_pct": 0.0, "cost_pct": 0.0, "churn": False}


def _apply_row_levers(df, levers, opex_pct=0.0, opex_abs=0.0):
    d = df.copy()
    if "price" not in d.columns or d["price"].isna().all():
        d["price"] = np.where(d.get("volume", pd.Series(np.nan, index=d.index)) != 0, d["revenue"] / d["volume"], np.nan)
    if "volume" in d.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            d["_uc"] = np.where(d["volume"] != 0, d["cost"] / d["volume"], np.nan)
    else:
        d["_uc"] = np.nan
    keep = pd.Series(True, index=d.index)
    for lv in levers:
        dim, scope, member = lv.get("dimension"), lv.get("scope", "member"), lv.get("member")
        if scope == "all" or member in ("*", None, "all"):
            sel = pd.Series(True, index=d.index)
        else:
            if dim not in d.columns: continue
            sel = d[dim] == member
        if lv.get("churn"):
            keep &= ~sel; continue
        if lv.get("volume_pct"): d.loc[sel, "volume"] = d.loc[sel, "volume"] * (1 + lv["volume_pct"])
        if lv.get("price_pct"): d.loc[sel, "price"] = d.loc[sel, "price"] * (1 + lv["price_pct"])
        if lv.get("cost_pct"): d.loc[sel, "_uc"] = d.loc[sel, "_uc"] * (1 + lv["cost_pct"])
    d = d[keep].copy()
    if "volume" in d.columns:
        d["revenue"] = d["volume"] * d["price"]; d["cost"] = d["volume"] * d["_uc"]; d["gross_margin"] = d["revenue"] - d["cost"]
    if "opex" in d.columns:
        if opex_pct: d["opex"] = d["opex"] * (1 + opex_pct)
        if opex_abs:
            tr = d["revenue"].sum()
            d["opex"] = d["opex"] + (opex_abs * (d["revenue"] / tr) if tr else opex_abs / max(len(d), 1))
    return d.drop(columns=["_uc"], errors="ignore")


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
        deltas[k] = {"delta": s - b, "unit": "pp"} if k == "gm_pct" else {"delta": s - b, "delta_pct": (s - b) / b if b else np.nan, "unit": "£"}
    return {"ok": True, "baseline": baseline, "scenario": scenario, "deltas": deltas,
            "scenario_df": scen_df, "_baseline_df": df, "levers": levers, "opex_pct": opex_pct, "opex_abs": opex_abs}


def members(df, dimension):
    if dimension not in df.columns: return []
    return df.groupby(dimension)["revenue"].sum().sort_values(ascending=False).index.tolist()


# --- Scenario bridges (baseline -> scenario), all reconcile ------------------ #
def _grid(df):
    keys = [k for k in ["customer", "product"] if k in df.columns]
    if not keys: return None, keys
    g = df.groupby(keys).agg(vol=("volume", "sum"), rev=("revenue", "sum"), cost=("cost", "sum")).reset_index()
    g["price"] = np.where(g["vol"] != 0, g["rev"] / g["vol"], np.nan)
    return g, keys


def scenario_sales_bridge(result):
    bdf, sdf = result["_baseline_df"], result["scenario_df"]
    base, cur = calc.sales(bdf), calc.sales(sdf)
    gb, keys = _grid(bdf); ga, _ = _grid(sdf)
    if gb is None or ga is None:
        return {"base_label": "Baseline", "base": base, "steps": [("Other / Residual", cur - base)], "current": cur, "reconciles": True}
    m = ga.merge(gb, on=keys, how="outer", suffixes=("_a", "_b"))
    a_p, b_p = m["rev_a"].notna(), m["rev_b"].notna(); both = a_p & b_p
    m = m.fillna(0); bop = base / gb["vol"].sum() if gb["vol"].sum() else 0.0
    vol = float(np.where(both, (m["vol_a"] - m["vol_b"]) * m["price_b"], 0).sum())
    price = float(np.where(both, (m["price_a"] - m["price_b"]) * m["vol_a"], 0).sum())
    mix = float(np.where(both, (m["price_b"] - bop) * (m["vol_a"] - m["vol_b"]), 0).sum()); vol -= mix
    new = float(np.where(a_p & ~b_p, m["rev_a"], 0).sum()); lost = float(np.where(~a_p & b_p, -m["rev_b"], 0).sum())
    steps = [("Volume", vol), ("Price", price), ("Mix", mix), ("New", new), ("Lost", lost)]
    steps.append(("Other / Residual", (cur - base) - sum(v for _, v in steps)))
    return {"base_label": "Baseline", "base": base, "steps": steps, "current": cur, "reconciles": abs(base + sum(v for _, v in steps) - cur) < TOL}


def scenario_gm_pct_bridge(result):
    bdf, sdf = result["_baseline_df"], result["scenario_df"]
    gm_b, gm_a = calc.gm_pct(bdf), calc.gm_pct(sdf)
    s_b, c_b, s_a, c_a = calc.sales(bdf), calc.cogs(bdf), calc.sales(sdf), calc.cogs(sdf)
    total = gm_a - gm_b
    if s_a and s_b:
        cost = -((c_a / s_a) - (c_b / s_b)); steps = [("Price / volume & mix", total - cost), ("Cost / raw material", cost)]
    else:
        steps = [("Price / volume & mix", total)]
    steps.append(("Other / Residual", total - sum(v for _, v in steps)))
    return {"base_label": "Baseline", "base": gm_b, "steps": steps, "current": gm_a, "unit": "pp", "reconciles": abs(gm_b + sum(v for _, v in steps) - gm_a) < 1e-6}


def scenario_ebitda_bridge(result):
    bdf, sdf = result["_baseline_df"], result["scenario_df"]
    e_b, e_a = calc.ebitda(bdf), calc.ebitda(sdf)
    s_b, s_a = calc.sales(bdf), calc.sales(sdf)
    gmr_b, gmr_a = calc.gm_pct(bdf), calc.gm_pct(sdf)
    ox_b, ox_a = calc.opex(bdf), calc.opex(sdf)
    steps = []
    if not any(np.isnan(x) for x in [s_a, s_b, gmr_b]): steps.append(("Sales / volume impact", (s_a - s_b) * gmr_b))
    if not any(np.isnan(x) for x in [gmr_a, gmr_b, s_a]): steps.append(("Gross margin impact", (gmr_a - gmr_b) * s_a))
    if not (np.isnan(ox_a) or np.isnan(ox_b)): steps.append(("OPEX impact", -(ox_a - ox_b)))
    steps.append(("Other / Residual", (e_a - e_b) - sum(v for _, v in steps)))
    return {"base_label": "Baseline", "base": e_b, "steps": steps, "current": e_a, "reconciles": abs(e_b + sum(v for _, v in steps) - e_a) < TOL}
