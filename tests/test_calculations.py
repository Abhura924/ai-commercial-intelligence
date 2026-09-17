"""Deterministic engine + scenario-agent + forecast-horizon tests."""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analytics import calculation_engine as calc
from analytics import bridges, attribution, budget as budget_engine, projection as pj
from analytics import scenarios as sc, forecast as fc
from ai import scenario_agent as agent
from demo.chemical_demo import build_canonical

DF = build_canonical()


# --- Core ---
def test_sales_is_sum_revenue():
    assert abs(calc.sales(DF) - DF["revenue"].sum()) < 1e-6

def test_gm_equals_sales_minus_cogs():
    assert abs(calc.gross_margin(DF) - (calc.sales(DF) - calc.cogs(DF))) < 1.0

def test_gm_pct_equals_gm_over_sales():
    assert abs(calc.gm_pct(DF) - calc.gross_margin(DF) / calc.sales(DF)) < 1e-9


# --- Bridges ---
def test_sales_bridge_reconciles():
    br = bridges.sales_bridge(DF, "budget")
    assert abs(br["base"] + sum(v for _, v in br["steps"]) - br["current"]) < bridges.TOL

def test_gm_pct_bridge_reconciles():
    br = bridges.gm_pct_bridge(DF, "budget")
    assert abs(br["base"] + sum(v for _, v in br["steps"]) - br["current"]) < 1e-6

def test_ebitda_bridge_reconciles():
    br = bridges.ebitda_bridge(DF, "budget")
    assert abs(br["base"] + sum(v for _, v in br["steps"]) - br["current"]) < bridges.TOL


# --- Attribution ---
def test_attribution_reconciles():
    a = attribution.attribute_sales(DF, "budget")
    assert abs(a["base"] + sum(a["effects"].values()) - a["current"]) < attribution.TOL


# --- Budget / projection ---
def test_budget_growth_identity():
    b = budget_engine.build_budget(DF, growth=0.06)
    assert abs(b["budget_rev"] - b["last_year_rev"] * 1.06) < 1.0

def test_projection_compounds():
    res = pj.project(DF, years=5, revenue_cagr=0.08)
    assert abs(res["exit_revenue"] - pj.base_year(DF)["revenue"] * 1.08 ** 5) < 1.0


# --- Scenario engine ---
def test_scenario_churn_reduces_sales_by_member_revenue():
    target = sc.members(DF, "customer")[0]
    member_rev = DF[DF["customer"] == target]["revenue"].sum()
    lever = sc.blank_lever(); lever.update({"dimension": "customer", "member": target, "churn": True})
    r = sc.run_scenario(DF, [lever])
    assert abs(r["deltas"]["sales"]["delta"] - (-member_rev)) < 1.0

def test_scenario_sales_bridge_reconciles():
    plan = agent.rule_based_parse("grow Customer A volume 15% and raise Product B price 5%", DF)
    r = sc.run_scenario(DF, plan["levers"], plan.get("opex_pct", 0.0), plan.get("opex_abs", 0.0))
    br = sc.scenario_sales_bridge(r)
    assert abs(br["base"] + sum(v for _, v in br["steps"]) - br["current"]) < sc.TOL

def test_scenario_ebitda_bridge_reconciles():
    plan = agent.rule_based_parse("company-wide price +3% and OPEX up £750k", DF)
    r = sc.run_scenario(DF, plan["levers"], plan.get("opex_pct", 0.0), plan.get("opex_abs", 0.0))
    br = sc.scenario_ebitda_bridge(r)
    assert abs(br["base"] + sum(v for _, v in br["steps"]) - br["current"]) < sc.TOL


# --- Agent NL parsing ---
def test_agent_parses_grow_customer():
    plan = agent.rule_based_parse("grow Customer A by 10%", DF)
    assert any(lv["member"] == "Customer A" and abs(lv["volume_pct"] - 0.10) < 1e-9 for lv in plan["levers"])

def test_agent_multi_driver_and_absolute_opex():
    t = "Customer A grows volume by 15%, we increase prices by 3%, raw material costs increase by 5%, and OPEX increases by £500k"
    plan = agent.rule_based_parse(t, DF)
    assert any(lv["member"] == "Customer A" and abs(lv["volume_pct"] - 0.15) < 1e-9 for lv in plan["levers"])
    assert any(lv.get("scope") == "all" and abs(lv.get("cost_pct", 0) - 0.05) < 1e-9 for lv in plan["levers"])
    assert abs(plan["opex_abs"] - 500000) < 1.0

def test_agent_sales_reduce_is_company_wide_no_clarify():
    plan = agent.rule_based_parse("sales to reduce by 3%", DF)
    assert any(lv.get("scope") == "all" and abs(lv.get("volume_pct", 0) + 0.03) < 1e-9 for lv in plan["levers"])
    assert agent.needs_clarification("sales to reduce by 3%", plan) is None

def test_agent_market_does_not_eat_driver_percent():
    plan = agent.rule_based_parse("sales to reduce by 0.2% due market change", DF)
    assert any(lv.get("scope") == "all" and abs(lv.get("volume_pct", 0) + 0.002) < 1e-9 for lv in plan["levers"])

def test_agent_bare_price_clarifies():
    res = agent.ask(DF, "increase price by 5%")
    assert res.get("clarification") is not None


# --- v0.4 Forecast horizon ---
def test_forecast_frame_builds_horizon():
    ff = fc.forecast_frame(DF, horizon=6)
    assert ff["ok"]
    # 6 distinct future months
    assert ff["frame"]["date"].dt.to_period("M").nunique() == 6

def test_horizon_extraction():
    assert agent.extract_horizon("grow Customer A 10% over the next 6 months") == 6
    assert agent.extract_horizon("raise prices 5% for next quarter") == 3
    assert agent.extract_horizon("grow Customer A 10% next year") == 12
    assert agent.extract_horizon("grow Customer A 10%") is None

def test_scenario_over_forecast_horizon_runs_and_reconciles():
    res = agent.ask(DF, "grow Customer A volume by 10% over the next 6 months")
    assert res["result"] is not None
    assert "forecast" in res["scope_label"].lower()
    br = sc.scenario_sales_bridge(res["result"])
    assert abs(br["base"] + sum(v for _, v in br["steps"]) - br["current"]) < sc.TOL
    # +10% on a customer over the forecast should raise sales
    assert res["result"]["deltas"]["sales"]["delta"] > 0


# --- v0.5 unified agent router ---
def test_router_classifies_scenario():
    from ai import agent_router as ar
    assert ar.classify_intent("what happens if Customer A grows 15% and prices +3%")[0] == "scenario"
    assert ar.classify_intent("lose Customer C and volume falls 10%")[0] == "scenario"

def test_router_classifies_dashboard():
    from ai import agent_router as ar
    assert ar.classify_intent("build me a board dashboard with KPIs and a sales bridge")[0] == "dashboard"
    assert ar.classify_intent("I have a board meeting in 15 minutes, show KPIs and GM%")[0] == "dashboard"

def test_router_classifies_question():
    from ai import agent_router as ar
    assert ar.classify_intent("why is sales below budget?")[0] == "question"

def test_router_route_scenario_runs_engine():
    from ai import agent_router as ar
    r = ar.route(DF, "grow Customer A volume by 10% over the next 6 months")
    assert r["kind"] == "scenario"
    assert r["result"]["deltas"]["sales"]["delta"] > 0
    assert "forecast" in r["scope_label"].lower()

def test_router_route_dashboard_returns_spec():
    from ai import agent_router as ar
    r = ar.route(DF, "board pack: 5 KPIs, sales bridge and the top 3 issues")
    assert r["kind"] == "dashboard"
    assert "kpis" in r["spec"]

def test_router_bare_price_clarifies():
    from ai import agent_router as ar
    r = ar.route(DF, "increase price 5%")
    assert r.get("clarification") is not None
