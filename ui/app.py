"""
AI Commercial Intelligence — v0.4 (Streamlit prototype UI)
=========================================================
Presentation only. All numbers from the deterministic engine.
v0.4 adds: (a) forecast-horizon selector so a scenario runs over a future period,
and (b) a natural-language Board Builder.
Run from project root:  streamlit run ui/app.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from analytics import calculation_engine as calc
from analytics import bridges, forecast as fc_engine, projection as proj_engine
from analytics import attribution as attr_engine, budget as budget_engine, scenarios as scen_engine
from ai import analyst, scenario_agent as agent, dashboard_builder as dbuilder, agent_router
from ai.llm_provider import get_provider
from demo.chemical_demo import build_canonical
from ui import theme, charts, pipeline
from ui.theme import kpi_card, panel_header, fmt_m, fmt_signed_m, fmt_pct, fmt_pp

st.set_page_config(page_title="AI Commercial Intelligence", layout="wide", page_icon="📈", initial_sidebar_state="expanded")
theme.inject_css()

if "canonical" not in st.session_state:
    st.session_state.canonical = build_canonical(); st.session_state.source_label = "Demo — chemical manufacturing"
if "base" not in st.session_state:
    st.session_state.base = "budget"
for k, d in [("sc_history", []), ("sc_horizon", "Current actuals"), ("agent_last", None)]:
    if k not in st.session_state:
        st.session_state[k] = d

DF = st.session_state.canonical
provider = get_provider()

NAV = {"Executive Dashboard": "📊", "AI Analyst Agent": "🤖", "Sales & Drivers": "🔀",
       "Forecast": "📈", "Budget": "🎯", "5-Year Projection": "🗓",
       "Bridges": "🧮", "Data & Quality": "📥"}
st.sidebar.markdown('<div class="mer-brand">AI Commercial<br>Intelligence</div>'
                    '<div class="mer-brand-sub">FP&amp;A &amp; Commercial Analyst · v0.5</div>', unsafe_allow_html=True)
page = st.sidebar.radio("nav", [f"{v}  {k}" for k, v in NAV.items()], label_visibility="collapsed").split("  ", 1)[1]

st.sidebar.markdown("---")
st.session_state.base = st.sidebar.radio("Compare against", ["budget", "prior_year"],
    format_func=lambda x: "Budget" if x == "budget" else "Prior year", horizontal=True)
BASE = st.session_state.base
if provider.available():
    st.sidebar.success(f"AI: {provider.name} ✔")
else:
    st.sidebar.info("AI off — parsing, commentary & board-building are rule-based & fully grounded. "
                    "Set ANTHROPIC_API_KEY / AZURE_OPENAI_* to enable LLM understanding.")
st.sidebar.markdown('<div class="mer-side-foot">Deterministic engine · every bridge reconciles · AI never invents numbers</div>', unsafe_allow_html=True)


def _tone(v, gp=True):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v == 0: return "flat"
    return ("up" if v > 0 else "down") if gp else ("down" if v > 0 else "up")


# =========================================================================== #
# Reusable dashboard blocks (used by Executive Dashboard AND Board Builder)
# =========================================================================== #
def block_kpis(df):
    k = calc.kpi_block(df); c = st.columns(5); s = k["sales"]
    c[0].markdown(kpi_card("Sales", fmt_m(s["value"]), delta=fmt_pct(s["variance_pct"]) + " vs Budget",
        tone=_tone(s["variance"]), delta2=(fmt_pct(s["py_pct"]) + " vs PY") if s["py_pct"] is not None else None, tone2=_tone(s["py_pct"])), unsafe_allow_html=True)
    gm = k["gross_margin"]
    c[1].markdown(kpi_card("Gross Margin", fmt_m(gm["value"]), delta=fmt_pct(gm["variance_pct"]) + " vs Budget",
        tone=_tone(gm["variance"]), delta2=(fmt_pct(gm["py_pct"]) + " vs PY") if gm["py_pct"] is not None else None, tone2=_tone(gm["py_pct"])), unsafe_allow_html=True)
    gp = k["gm_pct"]
    c[2].markdown(kpi_card("GM %", f"{gp['value']*100:.1f}%" if not np.isnan(gp['value']) else "—", delta=fmt_pp(gp["variance"]) + " vs Budget",
        tone=_tone(gp["variance"]), delta2=(fmt_pp(gp["py_pct"]) + " vs PY") if gp["py_pct"] is not None else None, tone2=_tone(gp["py_pct"])), unsafe_allow_html=True)
    eb = k["ebitda"]
    c[3].markdown(kpi_card("EBITDA", fmt_m(eb["value"]), delta=fmt_pct(eb["variance_pct"]) + " vs Budget",
        tone=_tone(eb["variance"]), delta2=(fmt_pct(eb["py_pct"]) + " vs PY") if eb["py_pct"] is not None else None, tone2=_tone(eb["py_pct"])), unsafe_allow_html=True)
    svb = k["sales_vs_budget"]
    c[4].markdown(kpi_card("Sales vs Budget", fmt_signed_m(svb["value"]), delta=fmt_pct(svb["variance_pct"]), tone=_tone(svb["value"])), unsafe_allow_html=True)

def block_sales_trend(df):
    panel_header("Sales performance", "Actual vs Budget vs Prior Year (monthly)")
    st.plotly_chart(charts.sales_trend(calc.timeseries(df, "M")), use_container_width=True)

def block_gm_trend(df):
    panel_header("GM% trend", "Actual vs Budget vs Prior Year")
    st.plotly_chart(charts.gm_trend(calc.timeseries(df, "M")), use_container_width=True)

def block_sales_bridge(df):
    panel_header("Sales bridge", f"{'Budget' if BASE=='budget' else 'Prior year'} → Actual")
    sb = bridges.sales_bridge(df, BASE); st.plotly_chart(charts.waterfall(sb), use_container_width=True)
    st.caption(("✅ reconciles" if sb["reconciles"] else "⚠️") + f" · {fmt_m(sb['base'])} → {fmt_m(sb['current'])}")

def block_gm_bridge(df):
    panel_header("GM% bridge", "Percentage-point decomposition")
    st.plotly_chart(charts.waterfall(bridges.gm_pct_bridge(df, BASE), unit="pp"), use_container_width=True)

def block_ebitda_bridge(df):
    panel_header("EBITDA bridge", "Sales · Gross margin · OPEX")
    eb = bridges.ebitda_bridge(df, BASE); st.plotly_chart(charts.waterfall(eb), use_container_width=True)

def block_drivers(df):
    panel_header("Commercial drivers", f"Biggest contributors vs {'budget' if BASE=='budget' else 'prior year'}")
    dl, dr = st.columns(2)
    with dl: st.markdown("**By customer**"); st.plotly_chart(charts.driver_bars(bridges.driver_ranking(df, "customer", BASE)), use_container_width=True)
    with dr: st.markdown("**By product**"); st.plotly_chart(charts.driver_bars(bridges.driver_ranking(df, "product", BASE)), use_container_width=True)

def block_risks_opps(df):
    cust = bridges.driver_ranking(df, "customer", BASE)
    rc, oc = st.columns(2)
    with rc:
        st.markdown("**Risks**")
        for name, v in cust["negative"][:3]: st.markdown(f"- {name}: {fmt_signed_m(v)}")
    with oc:
        st.markdown("**Opportunities**")
        for name, v in cust["positive"][:3]: st.markdown(f"- {name}: {fmt_signed_m(v)}")

def block_commentary(df):
    panel_header("AI Management Insights", "Grounded in the calculation engine — figures are calculated, not generated")
    comm = analyst.commentary(df, BASE)
    if comm.get("narrative"):
        st.markdown(f'<div class="mer-card">{comm["narrative"]}</div>', unsafe_allow_html=True)
        st.caption(f"Narrated by {comm['source']}")
    else:
        st.markdown(f'<div class="mer-card mer-what"><b>What.</b> {comm["what"]}</div>', unsafe_allow_html=True); st.write("")
        st.markdown(f'<div class="mer-card mer-why"><b>Why.</b> {comm["why"]}</div>', unsafe_allow_html=True); st.write("")
        st.markdown(f'<div class="mer-card mer-sowhat"><b>So what.</b> {comm["so_what"]}</div>', unsafe_allow_html=True)

def block_forecast(df):
    panel_header("Forecast", "Base / Upside / Downside — driver-based")
    st.plotly_chart(charts.scenario_chart(fc_engine.scenario_bundle(df, 6)), use_container_width=True)

BLOCK_FN = {"kpis": block_kpis, "sales_trend": block_sales_trend, "gm_trend": block_gm_trend,
            "sales_bridge": block_sales_bridge, "gm_bridge": block_gm_bridge, "ebitda_bridge": block_ebitda_bridge,
            "drivers": block_drivers, "risks_opps": block_risks_opps, "commentary": block_commentary, "forecast": block_forecast}


# =========================================================================== #
def page_dashboard():
    st.markdown(f'<div class="mer-topbar"><h1>Executive Dashboard</h1><span class="mer-pill">{st.session_state.source_label}</span></div>', unsafe_allow_html=True)
    block_kpis(DF); st.write(""); st.write("")
    l, r = st.columns(2)
    with l: block_sales_trend(DF)
    with r: block_gm_trend(DF)
    st.write(""); block_drivers(DF)
    st.write(""); block_commentary(DF)


# =========================================================================== #
def page_sales():
    st.markdown('<h1>Sales &amp; Drivers</h1>', unsafe_allow_html=True)
    st.markdown('<p class="mer-note">Advanced attribution: Volume · Price · Mix · New · Lost · Other. Fully reconciles, with drill-through.</p>', unsafe_allow_html=True)
    attr = attr_engine.attribute_sales(DF, BASE)
    if not attr.get("ok"): st.warning(attr.get("reason", "Cannot attribute.")); return
    panel_header("Sales attribution bridge", f"{attr['base_label']} → Actual")
    st.plotly_chart(charts.waterfall(attr_engine.bridge_steps(attr)), use_container_width=True)
    st.caption(f"{attr['base_label']} {fmt_m(attr['base'])} + effects = Actual {fmt_m(attr['current'])} · " + ("✅ reconciles" if attr["reconciles"] else "⚠️"))
    st.write(""); panel_header("Drill-through", "Pick an effect and dimension to see who / what drove it")
    c = st.columns([1, 1, 2]); effect = c[0].selectbox("Effect", ["Volume", "Price", "Mix", "New", "Lost"]); by = c[1].selectbox("By", ["customer", "product"])
    dd = attr_engine.drilldown(attr, effect, by)
    st.info("No breakdown.") if dd.empty else st.plotly_chart(charts.effect_drill_bars(dd, effect, by), use_container_width=True)


# =========================================================================== #
# AI SCENARIO ANALYST (with v0.4 forecast-horizon selector)
# =========================================================================== #
HORIZONS = {"Current actuals": None, "Next 3 months": 3, "Next 6 months": 6, "Next 12 months": 12}


def _kpi_impact(result):
    b, s, d = result["baseline"], result["scenario"], result["deltas"]; k = st.columns(4)
    k[0].markdown(kpi_card("Sales", fmt_m(s["sales"]), sub=f"was {fmt_m(b['sales'])}", delta=fmt_signed_m(d["sales"]["delta"]) + f" ({fmt_pct(d['sales']['delta_pct'])})", tone=_tone(d["sales"]["delta"])), unsafe_allow_html=True)
    k[1].markdown(kpi_card("Gross Margin", fmt_m(s["gross_margin"]), sub=f"was {fmt_m(b['gross_margin'])}", delta=fmt_signed_m(d["gross_margin"]["delta"]) + f" ({fmt_pct(d['gross_margin']['delta_pct'])})", tone=_tone(d["gross_margin"]["delta"])), unsafe_allow_html=True)
    k[2].markdown(kpi_card("GM %", f"{s['gm_pct']*100:.1f}%", sub=f"was {b['gm_pct']*100:.1f}%", delta=fmt_pp(d["gm_pct"]["delta"]), tone=_tone(d["gm_pct"]["delta"])), unsafe_allow_html=True)
    k[3].markdown(kpi_card("EBITDA", fmt_m(s["ebitda"]), sub=f"was {fmt_m(b['ebitda'])}", delta=fmt_signed_m(d["ebitda"]["delta"]) + f" ({fmt_pct(d['ebitda']['delta_pct'])})", tone=_tone(d["ebitda"]["delta"])), unsafe_allow_html=True)


def _render_scenario(result, scope, source, answer, understood, plan):
    pill = f'<span class="mer-pill mer-pill-amber">over {scope}</span>'
    st.markdown(f'<div class="mer-topbar"><div class="mer-panel-title">Scenario impact</div>{pill}</div>', unsafe_allow_html=True)
    st.caption(f"Baseline = {scope}; all figures recomputed by the deterministic engine")
    _kpi_impact(result)
    st.write("")
    c = st.columns([1.2, 1])
    with c[0]:
        panel_header("Baseline vs scenario", "Sales · Gross Margin · EBITDA")
        st.plotly_chart(charts.scenario_compare_bars(result["baseline"], result["scenario"], lbl_a="Baseline"), use_container_width=True)
    with c[1]:
        panel_header("EBITDA driver bridge", "Baseline → scenario")
        st.plotly_chart(charts.waterfall(scen_engine.scenario_ebitda_bridge(result)), use_container_width=True)
    st.write("")
    panel_header("Bridges", "Sales & GM% — reconcile exactly")
    bc = st.columns(2); sb = scen_engine.scenario_sales_bridge(result)
    with bc[0]:
        st.markdown("**Sales bridge**"); st.plotly_chart(charts.waterfall(sb), use_container_width=True)
        st.caption(("✅ reconciles" if sb["reconciles"] else "⚠️") + f" · {fmt_m(sb['base'])} → {fmt_m(sb['current'])}")
    gb = scen_engine.scenario_gm_pct_bridge(result)
    with bc[1]:
        st.markdown("**GM% bridge**"); st.plotly_chart(charts.waterfall(gb, unit="pp"), use_container_width=True)
        st.caption("✅ reconciles" if gb["reconciles"] else "⚠️")
    st.write("")
    panel_header("AI management commentary", "Explains only the deterministic scenario results")
    st.markdown(f'<div class="mer-card mer-why">{answer or ""}</div>', unsafe_allow_html=True)
    st.caption(f"Source: {source} · grounded in calculated deltas")
    # horizon quick-switch + save
    st.write("")
    hc = st.columns([2, 1])
    new_hz = hc[0].radio("Re-run this scenario over", list(HORIZONS.keys()),
                         index=list(HORIZONS.keys()).index(st.session_state.sc_horizon), horizontal=True, key="agent_hz")
    if hc[1].button("🔁 Apply horizon"):
        st.session_state.sc_horizon = new_hz
        plan["horizon"] = HORIZONS[new_hz]
        run = agent.run_plan(DF, plan, BASE)
        st.session_state.agent_last = {"kind": "scenario", "result": run["result"], "understood": run["understood"],
                                       "answer": run["answer"], "answer_source": run["source"], "scope_label": run["scope_label"], "plan": plan}
        st.rerun()
    sv = st.columns([2, 1])
    name = sv[0].text_input("Scenario name", value=understood or "Scenario", label_visibility="collapsed", key="agent_scname")
    if sv[1].button("💾 Save scenario", use_container_width=True):
        d = result["deltas"]
        st.session_state.sc_history.append({"Name": name, "Scope": scope, "Assumptions": understood,
            "Sales Δ": d["sales"]["delta"], "GM Δ": d["gross_margin"]["delta"], "EBITDA Δ": d["ebitda"]["delta"]})
        st.success(f"Saved “{name}”.")


def _render_dashboard(chosen, source):
    panel_header("Dashboard assembled", f"Interpreted by {source} · blocks: " + ", ".join(chosen))
    st.markdown(f'<div class="mer-topbar"><h1>Board Pack</h1><span class="mer-pill">{st.session_state.source_label}</span></div>', unsafe_allow_html=True)
    chart_blocks = {"sales_trend", "gm_trend", "sales_bridge", "gm_bridge", "ebitda_bridge", "forecast"}
    i = 0
    while i < len(chosen):
        b = chosen[i]
        if b == "kpis":
            BLOCK_FN[b](DF); st.write(""); i += 1; continue
        if b in chart_blocks and i + 1 < len(chosen) and chosen[i + 1] in chart_blocks:
            c = st.columns(2)
            with c[0]: BLOCK_FN[b](DF)
            with c[1]: BLOCK_FN[chosen[i + 1]](DF)
            st.write(""); i += 2; continue
        BLOCK_FN[b](DF); st.write(""); i += 1


def page_agent():
    st.markdown('<h1>AI Analyst Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="mer-note">One agent for everything — ask a question, model a what-if scenario, '
                'or build a board dashboard. Just describe what you need; the agent works out whether to '
                'analyse, simulate, or visualise. The deterministic engine does every calculation.</p>', unsafe_allow_html=True)

    with st.form("agent_form", clear_on_submit=False):
        cc = st.columns([4, 1.3])
        q = cc[0].text_input("What can I do for you today?",
                             placeholder="e.g. “build me a board pack with 5 KPIs and a sales bridge”, or “what if prices rise 5% over the next 6 months?”, or “why is sales below budget?”",
                             label_visibility="collapsed")
        hz_label = cc[1].selectbox("Apply over", list(HORIZONS.keys()),
                                   index=list(HORIZONS.keys()).index(st.session_state.sc_horizon), label_visibility="collapsed")
        go = st.form_submit_button("✨  Ask the agent")
    st.session_state.sc_horizon = hz_label

    # Example chips spanning all three capabilities
    ex = st.columns(3)
    examples = [("📊 Board pack: 5 KPIs, sales bridge and the top 3 issues", None),
                ("🧪 What if Customer A grows 15%, prices +3%, cost +5%, OPEX +£500k", HORIZONS[hz_label]),
                ("💬 Why is sales below budget?", None)]
    triggered = None
    for i, (label, _) in enumerate(examples):
        if ex[i].button(label, use_container_width=True):
            triggered = label.split(" ", 1)[1]  # strip emoji

    request = (q.strip() if (go and q.strip()) else triggered)
    if request:
        res = agent_router.route(DF, request, BASE)
        # if it's a scenario, honour the horizon selector
        if res["kind"] == "scenario" and res.get("result") is not None and HORIZONS[hz_label] is not None:
            res["plan"]["horizon"] = HORIZONS[hz_label]
            run = agent.run_plan(DF, res["plan"], BASE)
            res.update({"result": run["result"], "understood": run["understood"], "answer": run["answer"],
                        "answer_source": run["source"], "scope_label": run["scope_label"]})
        st.session_state.agent_last = res

    res = st.session_state.get("agent_last")
    if not res:
        return

    st.write("")
    # a small routed-intent indicator
    kind_label = {"scenario": "🧪 Scenario", "dashboard": "📊 Dashboard", "question": "💬 Answer"}.get(res["kind"], res["kind"])
    st.markdown(f'<span class="mer-badge">{kind_label}</span> '
                f'<span class="mer-note">routed by {res.get("intent_source","rule-based")} · “{res.get("request","")}”</span>',
                unsafe_allow_html=True)
    st.write("")

    if res.get("clarification"):
        st.warning("🤔 " + res["clarification"]); return

    if res["kind"] == "scenario":
        plan = res["plan"]
        scope = res.get("scope_label", "current actuals")
        panel_header("Scenario understood", f"Interpreted by {res.get('answer_source','rule-based')} · applied over {scope}")
        st.markdown('<div class="mer-understood">', unsafe_allow_html=True)
        rows = agent.assumptions_table(plan)
        st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame([{"Scope": "—", "Driver": "—", "Change": "—"}]), hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
        _render_scenario(res["result"], scope, res.get("answer_source", "rule-based"), res.get("answer"), res.get("understood"), plan)
        if st.session_state.sc_history:
            st.write(""); panel_header("Scenario history", "Saved scenarios")
            hv = pd.DataFrame([{"Name": h["Name"], "Scope": h["Scope"], "Assumptions": h["Assumptions"],
                                "Sales Δ": fmt_signed_m(h["Sales Δ"]), "GM Δ": fmt_signed_m(h["GM Δ"]), "EBITDA Δ": fmt_signed_m(h["EBITDA Δ"])} for h in st.session_state.sc_history])
            st.dataframe(hv, hide_index=True, use_container_width=True)
            if st.button("🗑 Clear history"): st.session_state.sc_history = []; st.rerun()

    elif res["kind"] == "dashboard":
        _render_dashboard(res["spec"], res.get("spec_source", "rule-based"))

    elif res["kind"] == "question":
        panel_header("Answer", "Grounded in the calculation engine")
        st.markdown(f'<div class="mer-card mer-why">{res["answer"]}</div>', unsafe_allow_html=True)
        st.caption(f"Source: {res.get('answer_source','rule-based')} · figures from deterministic engine")


# =========================================================================== #
def page_forecast():
    st.markdown('<h1>Forecast</h1>', unsafe_allow_html=True)
    horizon = st.columns([1, 3])[0].selectbox("Horizon (months)", [3, 6, 9, 12], index=1)
    bundle = fc_engine.scenario_bundle(DF, horizon); base_fc = bundle["Base"]
    if not base_fc.get("ok"): st.warning(base_fc.get("reason", "Cannot forecast.")); return
    k = st.columns(3)
    k[0].markdown(kpi_card("Base forecast (next period)", fmt_m(base_fc["values"][0]), tone="flat"), unsafe_allow_html=True)
    k[1].markdown(kpi_card(f"Base forecast ({horizon}m total)", fmt_m(base_fc["forecast_total"]), tone="flat"), unsafe_allow_html=True)
    k[2].markdown(kpi_card("Blended monthly growth", fmt_pct(base_fc["blended_monthly_growth"]), tone=_tone(base_fc["blended_monthly_growth"])), unsafe_allow_html=True)
    st.write(""); st.write(""); panel_header("Scenario forecast", "Base / Upside / Downside — driver-based, explainable")
    st.plotly_chart(charts.scenario_chart(bundle), use_container_width=True)
    st.write(""); panel_header("Why is this the forecast?", "Additive driver contributions (monthly)")
    st.dataframe(base_fc["explanation"], hide_index=True, use_container_width=True)
    st.caption("💡 Tip: in the AI Scenario Analyst you can apply a what-if *over* this forecast horizon.")


# =========================================================================== #
def page_budget():
    st.markdown('<h1>Budget</h1>', unsafe_allow_html=True)
    st.markdown('<p class="mer-note">Predictive next-year budget, bottom-up from the last 12 months.</p>', unsafe_allow_html=True)
    c = st.columns(3)
    growth = c[0].slider("Revenue growth", -0.10, 0.30, 0.06, 0.005, format="%.1f%%")
    gm = c[1].slider("Target gross margin", 0.10, 0.85, float(round(calc.gm_pct(DF), 2)) if not np.isnan(calc.gm_pct(DF)) else 0.35, 0.005, format="%.0f%%")
    infl = c[2].slider("Cost inflation on COGS", 0.0, 0.15, 0.03, 0.005, format="%.1f%%")
    b = budget_engine.build_budget(DF, growth, gm, infl)
    if not b.get("ok"): st.warning(b.get("reason")); return
    ny = b["next_year"]; k = st.columns(3)
    k[0].markdown(kpi_card(f"{ny} budgeted revenue", fmt_m(b["budget_rev"]), delta=fmt_pct(growth) + " vs last 12m", tone=_tone(growth)), unsafe_allow_html=True)
    k[1].markdown(kpi_card(f"{ny} gross margin", f"{gm*100:.1f}%", delta=fmt_m(b["budget_gm_pounds"]) + " GM£", tone="flat"), unsafe_allow_html=True)
    k[2].markdown(kpi_card(f"{ny} budgeted COGS", fmt_m(b["budget_cogs"]), delta=fmt_pct(infl) + " cost inflation", tone="flat"), unsafe_allow_html=True)
    st.write(""); st.write(""); panel_header(f"{ny} budget — quarterly phasing", "Seasonalised from the last four quarters")
    st.plotly_chart(charts.budget_phasing(b["q_budget"], b["last_year_actual_quarters"]), use_container_width=True)


# =========================================================================== #
def page_projection():
    st.markdown('<h1>5-Year Projection</h1>', unsafe_allow_html=True)
    b = proj_engine.base_year(DF)
    if not b.get("ok"): st.warning(b.get("reason")); return
    seed = proj_engine.suggested_assumptions(DF); c = st.columns(4)
    years = c[0].selectbox("Horizon (years)", [3, 5, 7, 10], index=1)
    cagr = c[1].slider("Revenue growth (CAGR)", -0.15, 0.35, float(seed["revenue_cagr"]), 0.005, format="%.1f%%")
    gm_drift = c[2].slider("Annual GM% change (pp)", -0.03, 0.03, 0.0, 0.0025, format="%.2f")
    opex_eff = c[3].slider("Annual OPEX efficiency (pp of rev)", 0.0, 0.02, 0.0, 0.0025, format="%.2f")
    res = proj_engine.project(DF, years, cagr, gm_drift, opex_eff); bundle = proj_engine.scenario_bundle(DF, years, cagr, gm_drift, opex_eff); proj = res["projection"]
    st.write(""); k = st.columns(4)
    k[0].markdown(kpi_card(f"{proj['year'].iloc[-1]} revenue", fmt_m(res["exit_revenue"]), delta=fmt_pct(cagr) + " CAGR", tone=_tone(cagr)), unsafe_allow_html=True)
    k[1].markdown(kpi_card(f"{years}-yr cumulative revenue", fmt_m(res["cumulative_revenue"]), tone="flat"), unsafe_allow_html=True)
    k[2].markdown(kpi_card(f"{proj['year'].iloc[-1]} GM%", f"{res['exit_gm_pct']*100:.1f}%", delta=fmt_pp(gm_drift * years) + " vs base", tone=_tone(gm_drift)), unsafe_allow_html=True)
    k[3].markdown(kpi_card(f"{proj['year'].iloc[-1]} EBITDA", fmt_m(res["exit_ebitda"]), delta=fmt_m(res["cumulative_ebitda"]) + f" cum.", tone="flat"), unsafe_allow_html=True)
    st.write(""); st.write(""); panel_header("Revenue & EBITDA projection", f"Base case · from {b['base_year']} actuals of {fmt_m(b['revenue'])}")
    st.plotly_chart(charts.projection_chart(proj), use_container_width=True)
    st.write(""); panel_header("Scenario range", "Base / Upside / Downside revenue paths")
    st.plotly_chart(charts.projection_scenarios(bundle), use_container_width=True)


# =========================================================================== #
def page_bridges():
    st.markdown('<h1>Bridges</h1>', unsafe_allow_html=True)
    st.markdown('<p class="mer-note">Every bridge reconciles: base + drivers = current. Residual shown explicitly.</p>', unsafe_allow_html=True)
    block_sales_bridge(DF); st.write(""); block_gm_bridge(DF); st.write(""); block_ebitda_bridge(DF)


# =========================================================================== #
def page_data():
    st.markdown('<h1>Data &amp; Quality</h1>', unsafe_allow_html=True)
    st.markdown('<p class="mer-note">Upload Excel/CSV — no rigid template. The system infers columns, standardises, and runs quality checks.</p>', unsafe_allow_html=True)
    up = st.file_uploader("Upload sales & margin data", type=["xlsx", "xls", "csv"])
    if st.button("↺ Load the built-in chemical demo dataset"):
        st.session_state.canonical = build_canonical(); st.session_state.source_label = "Demo — chemical manufacturing"
        st.session_state.agent_last = None
        st.success("Loaded the demo dataset. Open the Executive Dashboard →")
    if up is not None:
        sheets = pipeline.read_upload(up); sheet = st.selectbox("Sheet", list(sheets.keys())); raw = sheets[sheet]
        st.markdown("**Preview**"); st.dataframe(raw.head(6), use_container_width=True)
        result = pipeline.process(raw, filename=up.name)
        st.write(""); panel_header("Data understanding", "Inferred semantic mapping")
        st.dataframe(result["dictionary"], hide_index=True, use_container_width=True)
        for note in result["derivations"]: st.caption("• " + note)
        st.write(""); panel_header("Data quality", "Nothing important is silently ignored")
        for f in result["findings"]:
            icon = {"ok": "✓", "warn": "⚠", "error": "✗"}[f["status"]]
            (st.success if f["status"] == "ok" else st.warning if f["status"] == "warn" else st.error)(f"{icon} {f['message']}")
        if st.button("✅ Use this dataset for analysis"):
            st.session_state.canonical = result["canonical"]; st.session_state.source_label = f"Uploaded — {up.name}"
            st.session_state.agent_last = None
            st.success(f"Loaded {len(result['canonical'])} rows. Open the Executive Dashboard →")


{"Executive Dashboard": page_dashboard, "AI Analyst Agent": page_agent, "Sales & Drivers": page_sales,
 "Forecast": page_forecast, "Budget": page_budget, "5-Year Projection": page_projection,
 "Bridges": page_bridges, "Data & Quality": page_data}[page]()
