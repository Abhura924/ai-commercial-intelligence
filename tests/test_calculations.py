"""Deterministic engine + scenario-agent tests."""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analytics import calculation_engine as calc
from analytics import bridges, attribution, budget as budget_engine, projection as pj, scenarios as sc
from ai import scenario_agent as agent
from demo.chemical_demo import build_canonical

DF = build_canonical()


def test_sales_is_sum_revenue():
    assert abs(calc.sales(DF) - DF["revenue"].sum()) < 1e-6

def test_gm_equals_sales_minus_cogs():
    assert abs(calc.gross_margin(DF) - (calc.sales(DF) - calc.cogs(DF))) < 1.0

def test_sales_bridge_reconciles():
    br = bridges.sales_bridge(DF, "budget")
    assert abs(br["base"] + sum(v for _, v in br["steps"]) - br["current"]) < bridges.TOL

def test_gm_pct_bridge_reconciles():
    br = bridges.gm_pct_bridge(DF, "budget")
    assert abs(br["base"] + sum(v for _, v in br["steps"]) - br["current"]) < 1e-6

def test_ebitda_bridge_reconciles():
    br = bridges.ebitda_bridge(DF, "budget")
    assert abs(br["base"] + sum(v for _, v in br["steps"]) - br["current"]) < bridges.TOL

def test_attribution_reconciles():
    a = attribution.attribute_sales(DF, "budget")
    assert abs(a["base"] + sum(a["effects"].values()) - a["current"]) < attribution.TOL

def test_attribution_lines_sum_to_totals():
    a = attribution.attribute_sales(DF, "budget")
    for eff in ["Volume", "Price", "Mix", "New", "Lost"]:
        assert abs(a["lines"][eff].sum() - a["effects"][eff]) < 1.0

def test_budget_growth_identity():
    b = budget_engine.build_budget(DF, growth=0.06)
    assert abs(b["budget_rev"] - b["last_year_rev"] * 1.06) < 1.0

def test_projection_compounds():
    res = pj.project(DF, years=5, revenue_cagr=0.08)
    assert abs(res["exit_revenue"] - pj.base_year(DF)["revenue"] * 1.08 ** 5) < 1.0

def test_scenario_churn_reduces_sales_by_member_revenue():
    members = sc.members(DF, "customer"); target = members[0]
    member_rev = DF[DF["customer"] == target]["revenue"].sum()
    lever = sc.blank_lever(); lever.update({"dimension": "customer", "member": target, "churn": True})
    r = sc.run_scenario(DF, [lever])
    assert abs(r["deltas"]["sales"]["delta"] - (-member_rev)) < 1.0


# --- v0.3.1 AI Scenario Agent: natural language -> levers ---
def test_agent_parses_grow_customer():
    plan = agent.rule_based_parse("grow Customer A by 10%", DF)
    assert any(lv["member"] == "Customer A" and abs(lv["volume_pct"] - 0.10) < 1e-9 for lv in plan["levers"])

def test_agent_parses_churn():
    plan = agent.rule_based_parse("we lose Customer C next year", DF)
    assert any(lv["member"] == "Customer C" and lv["churn"] for lv in plan["levers"])

def test_agent_parses_price_uplift():
    plan = agent.rule_based_parse("raise Product B price by 5%", DF)
    assert any(lv["member"] == "Product B" and abs(lv["price_pct"] - 0.05) < 1e-9 for lv in plan["levers"])

def test_agent_end_to_end_runs_engine():
    res = agent.ask(DF, "grow Customer A by 10%")
    assert res["result"] is not None
    # +10% on the largest customer must raise sales & EBITDA (deterministic)
    assert res["result"]["deltas"]["sales"]["delta"] > 0
    assert res["result"]["deltas"]["ebitda"]["delta"] > 0

def test_agent_handles_unparseable():
    res = agent.ask(DF, "hello there")
    assert res["understood"] is None
    assert res["result"] is None


# --- v0.3.2 AI Scenario Analyst: NL extraction, absolute OPEX, bridges, clarify ---
def test_agent_parses_multi_driver_sentence():
    t = ("What happens if Customer A grows volume by 15%, we increase prices by 3%, "
         "raw material costs increase by 5%, and OPEX increases by £500k?")
    plan = agent.rule_based_parse(t, DF)
    # Customer A volume +15%
    assert any(lv["member"] == "Customer A" and abs(lv["volume_pct"] - 0.15) < 1e-9 for lv in plan["levers"])
    # a company-wide cost +5% lever
    assert any(lv.get("scope") == "all" and abs(lv.get("cost_pct", 0) - 0.05) < 1e-9 for lv in plan["levers"])
    # absolute OPEX +£500k
    assert abs(plan["opex_abs"] - 500000) < 1.0

def test_agent_absolute_opex_reduces_ebitda():
    plan = agent.rule_based_parse("increase OPEX by £1m", DF)
    r = sc.run_scenario(DF, plan["levers"], plan.get("opex_pct", 0.0), plan.get("opex_abs", 0.0))
    assert abs(r["deltas"]["ebitda"]["delta"] - (-1_000_000)) < 1.0

def test_agent_clarifies_bare_price_change():
    res = agent.ask(DF, "increase price by 5%")
    assert res.get("clarification") is not None
    assert res["result"] is None

def test_scenario_sales_bridge_reconciles():
    plan = agent.rule_based_parse("grow Customer A volume 15% and raise Product B price 5%", DF)
    r = sc.run_scenario(DF, plan["levers"], plan.get("opex_pct", 0.0), plan.get("opex_abs", 0.0))
    br = sc.scenario_sales_bridge(r)
    assert abs(br["base"] + sum(v for _, v in br["steps"]) - br["current"]) < sc.TOL
    assert br["reconciles"]

def test_scenario_ebitda_bridge_reconciles():
    plan = agent.rule_based_parse("company-wide price +3% and OPEX up £750k", DF)
    r = sc.run_scenario(DF, plan["levers"], plan.get("opex_pct", 0.0), plan.get("opex_abs", 0.0))
    br = sc.scenario_ebitda_bridge(r)
    assert abs(br["base"] + sum(v for _, v in br["steps"]) - br["current"]) < sc.TOL

def test_scenario_gm_pct_bridge_reconciles():
    plan = agent.rule_based_parse("raw material costs rise 4%", DF)
    r = sc.run_scenario(DF, plan["levers"], plan.get("opex_pct", 0.0), plan.get("opex_abs", 0.0))
    br = sc.scenario_gm_pct_bridge(r)
    assert abs(br["base"] + sum(v for _, v in br["steps"]) - br["current"]) < 1e-6


# --- v0.3.3 parser regressions: company-wide sales/revenue, market vs driver ---
def test_agent_sales_reduce_is_company_wide():
    plan = agent.rule_based_parse("sales to reduce by 3%", DF)
    assert any(lv.get("scope") == "all" and abs(lv.get("volume_pct", 0) + 0.03) < 1e-9 for lv in plan["levers"])
    assert agent.needs_clarification("sales to reduce by 3%", plan) is None

def test_agent_market_does_not_eat_driver_percent():
    t = "sales to reduce by 0.2% due market change"
    plan = agent.rule_based_parse(t, DF)
    # the 0.2% must be a company-wide volume driver, NOT market growth
    assert any(lv.get("scope") == "all" and abs(lv.get("volume_pct", 0) + 0.002) < 1e-9 for lv in plan["levers"])

def test_agent_market_growth_when_adjacent():
    plan = agent.rule_based_parse("market grows 5% and our volume grows 2%", DF)
    assert abs((plan.get("market_growth") or 0) - 0.05) < 1e-9
    assert any(lv.get("scope") == "all" and abs(lv.get("volume_pct", 0) - 0.02) < 1e-9 for lv in plan["levers"])

def test_agent_bare_price_still_clarifies():
    res = agent.ask(DF, "increase price by 5%")
    assert res.get("clarification") is not None
