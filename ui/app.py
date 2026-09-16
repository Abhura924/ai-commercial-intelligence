"""
AI Commercial Intelligence — v0.3.2 (Streamlit prototype UI)
===========================================================
Presentation layer only. All numbers from the deterministic engine.
Scenario Lab is redesigned as a natural-language "AI Scenario Analyst".
Run from project root:  streamlit run ui/app.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from analytics import calculation_engine as calc
from analytics import bridges, forecast as fc_engine, projection as proj_engine
from analytics import attribution as attr_engine, budget as budget_engine, scenarios as scen_engine
from ai import analyst, scenario_agent as agent
from ai.llm_provider import get_provider
from demo.chemical_demo import build_canonical
from ui import theme, charts, pipeline
from ui.theme import kpi_card, panel_header, fmt_m, fmt_signed_m, fmt_pct, fmt_pp

st.set_page_config(page_title="AI Commercial Intelligence", layout="wide",
                   page_icon="📈", initial_sidebar_state="expanded")
theme.inject_css()

if "canonical" not in st.session_state:
    st.session_state.canonical = build_canonical()
    st.session_state.source_label = "Demo — chemical manufacturing"
if "base" not in st.session_state:
    st.session_state.base = "budget"
for key, default in [("sc_plan", None), ("sc_result", None), ("sc_understood", None),
                     ("sc_answer", None), ("sc_source", None), ("sc_clarify", None),
                     ("sc_history", []), ("sc_chat", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

DF = st.session_state.canonical
provider = get_provider()

NAV = {"Executive Dashboard": "📊", "Sales & Drivers": "🔀", "AI Scenario Analyst": "🧪",
       "Forecast": "📈", "Budget": "🎯", "5-Year Projection": "🗓", "Bridges": "🧮",
       "AI Analyst": "💬", "Data & Quality": "📥"}
st.sidebar.markdown('<div class="mer-brand">AI Commercial<br>Intelligence</div>'
                    '<div class="mer-brand-sub">FP&amp;A &amp; Commercial Analyst · v0.3</div>', unsafe_allow_html=True)
page = st.sidebar.radio("nav", [f"{v}  {k}" for k, v in NAV.items()], label_visibility="collapsed")
page = page.split("  ", 1)[1]

st.sidebar.markdown("---")
st.session_state.base = st.sidebar.radio("Compare against", ["budget", "prior_year"],
    format_func=lambda x: "Budget" if x == "budget" else "Prior year", horizontal=True)
BASE = st.session_state.base

if provider.available():
    st.sidebar.success(f"AI narration: {provider.name} ✔")
else:
    st.sidebar.info("AI narration off — parsing & commentary are rule-based & fully grounded. "
                    "Set ANTHROPIC_API_KEY / AZURE_OPENAI_* to enable LLM understanding.")
st.sidebar.markdown('<div class="mer-side-foot">Deterministic engine · every bridge '
                    'reconciles · AI never invents numbers</div>', unsafe_allow_html=True)


def _tone(v, gp=True):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v == 0:
        return "flat"
    return ("up" if v > 0 else "down") if gp else ("down" if v > 0 else "up")


# =========================================================================== #
def page_dashboard():
    k = calc.kpi_block(DF)
    st.markdown(f'<div class="mer-topbar"><h1>Executive Dashboard</h1>'
                f'<span class="mer-pill">{st.session_state.source_label}</span></div>', unsafe_allow_html=True)
    c = st.columns(5); s = k["sales"]
    c[0].markdown(kpi_card("Sales", fmt_m(s["value"]), delta=fmt_pct(s["variance_pct"]) + " vs Budget",
        tone=_tone(s["variance"]), delta2=(fmt_pct(s["py_pct"]) + " vs PY") if s["py_pct"] is not None else None, tone2=_tone(s["py_pct"])), unsafe_allow_html=True)
    gm = k["gross_margin"]
    c[1].markdown(kpi_card("Gross Margin", fmt_m(gm["value"]), delta=fmt_pct(gm["variance_pct"]) + " vs Budget",
        tone=_tone(gm["variance"]), delta2=(fmt_pct(gm["py_pct"]) + " vs PY") if gm["py_pct"] is not None else None, tone2=_tone(gm["py_pct"])), unsafe_allow_html=True)
    gp = k["gm_pct"]
    c[2].markdown(kpi_card("GM %", f"{gp['value']*100:.1f}%" if not np.isnan(gp['value']) else "—",
        delta=fmt_pp(gp["variance"]) + " vs Budget", tone=_tone(gp["variance"]),
        delta2=(fmt_pp(gp["py_pct"]) + " vs PY") if gp["py_pct"] is not None else None, tone2=_tone(gp["py_pct"])), unsafe_allow_html=True)
    eb = k["ebitda"]
    c[3].markdown(kpi_card("EBITDA", fmt_m(eb["value"]), delta=fmt_pct(eb["variance_pct"]) + " vs Budget",
        tone=_tone(eb["variance"]), delta2=(fmt_pct(eb["py_pct"]) + " vs PY") if eb["py_pct"] is not None else None, tone2=_tone(eb["py_pct"])), unsafe_allow_html=True)
    svb = k["sales_vs_budget"]
    c[4].markdown(kpi_card("Sales vs Budget", fmt_signed_m(svb["value"]), delta=fmt_pct(svb["variance_pct"]), tone=_tone(svb["value"])), unsafe_allow_html=True)

    st.write(""); st.write("")
    left, right = st.columns(2)
    with left:
        panel_header("Sales performance", "Actual vs Budget vs Prior Year (monthly)")
        st.plotly_chart(charts.sales_trend(calc.timeseries(DF, "M")), use_container_width=True)
    with right:
        panel_header("GM% trend", "Actual vs Budget vs Prior Year")
        st.plotly_chart(charts.gm_trend(calc.timeseries(DF, "M")), use_container_width=True)

    st.write("")
    panel_header("AI Management Insights", "Grounded in the calculation engine — figures are calculated, not generated")
    comm = analyst.commentary(DF, BASE)
    if comm.get("narrative"):
        st.markdown(f'<div class="mer-card">{comm["narrative"]}</div>', unsafe_allow_html=True)
        st.caption(f"Narrated by {comm['source']} · figures from deterministic engine")
    else:
        st.markdown(f'<div class="mer-card mer-what"><b>What.</b> {comm["what"]}</div>', unsafe_allow_html=True); st.write("")
        st.markdown(f'<div class="mer-card mer-why"><b>Why.</b> {comm["why"]}</div>', unsafe_allow_html=True); st.write("")
        st.markdown(f'<div class="mer-card mer-sowhat"><b>So what.</b> {comm["so_what"]}</div>', unsafe_allow_html=True)
        st.caption("Rule-based commentary (fully grounded). Configure an LLM for polished prose.")


# =========================================================================== #
def page_sales():
    st.markdown('<h1>Sales &amp; Drivers</h1>', unsafe_allow_html=True)
    st.markdown('<p class="mer-note">Advanced attribution: Volume · Price · Mix · New · Lost · Other. '
                'Fully reconciles, and every effect drills through to the customers and products behind it.</p>', unsafe_allow_html=True)
    attr = attr_engine.attribute_sales(DF, BASE)
    if not attr.get("ok"):
        st.warning(attr.get("reason", "Cannot attribute.")); return
    panel_header("Sales attribution bridge", f"{attr['base_label']} → Actual")
    st.plotly_chart(charts.waterfall(attr_engine.bridge_steps(attr)), use_container_width=True)
    st.caption(f"{attr['base_label']} {fmt_m(attr['base'])} + effects = Actual {fmt_m(attr['current'])} · "
               + ("✅ reconciles" if attr["reconciles"] else "⚠️ residual only"))
    st.write("")
    panel_header("Drill-through", "Pick an effect and dimension to see who / what drove it")
    c = st.columns([1, 1, 2])
    effect = c[0].selectbox("Effect", ["Volume", "Price", "Mix", "New", "Lost"])
    by = c[1].selectbox("By", ["customer", "product"])
    dd = attr_engine.drilldown(attr, effect, by)
    if dd.empty:
        st.info("No breakdown available for this effect/dimension.")
    else:
        st.plotly_chart(charts.effect_drill_bars(dd, effect, by), use_container_width=True)


# =========================================================================== #
# v0.3.2 — AI SCENARIO ANALYST (natural language)
# =========================================================================== #
def _kpi_impact_cards(result):
    b, s, d = result["baseline"], result["scenario"], result["deltas"]
    k = st.columns(4)
    k[0].markdown(kpi_card("Sales", fmt_m(s["sales"]), sub=f"was {fmt_m(b['sales'])}",
        delta=fmt_signed_m(d["sales"]["delta"]) + f" ({fmt_pct(d['sales']['delta_pct'])})", tone=_tone(d["sales"]["delta"])), unsafe_allow_html=True)
    k[1].markdown(kpi_card("Gross Margin", fmt_m(s["gross_margin"]), sub=f"was {fmt_m(b['gross_margin'])}",
        delta=fmt_signed_m(d["gross_margin"]["delta"]) + f" ({fmt_pct(d['gross_margin']['delta_pct'])})", tone=_tone(d["gross_margin"]["delta"])), unsafe_allow_html=True)
    k[2].markdown(kpi_card("GM %", f"{s['gm_pct']*100:.1f}%", sub=f"was {b['gm_pct']*100:.1f}%",
        delta=fmt_pp(d["gm_pct"]["delta"]), tone=_tone(d["gm_pct"]["delta"])), unsafe_allow_html=True)
    k[3].markdown(kpi_card("EBITDA", fmt_m(s["ebitda"]), sub=f"was {fmt_m(b['ebitda'])}",
        delta=fmt_signed_m(d["ebitda"]["delta"]) + f" ({fmt_pct(d['ebitda']['delta_pct'])})", tone=_tone(d["ebitda"]["delta"])), unsafe_allow_html=True)


def _run_and_store(question):
    """Parse the question, handle clarification, run, and store in session."""
    res = agent.ask(DF, question, BASE)
    st.session_state.sc_plan = res.get("plan")
    st.session_state.sc_understood = res.get("understood")
    st.session_state.sc_source = res.get("source")
    st.session_state.sc_clarify = res.get("clarification")
    st.session_state.sc_result = res.get("result")
    st.session_state.sc_answer = res.get("answer")


def _rerun_plan():
    plan = st.session_state.sc_plan
    if not plan:
        return
    run = agent.run_plan(DF, plan, BASE)
    st.session_state.sc_result = run["result"]
    st.session_state.sc_understood = run["understood"]
    st.session_state.sc_answer = run["answer"]
    st.session_state.sc_source = run["source"]
    st.session_state.sc_clarify = None


def page_scenarios():
    st.markdown('<h1>AI Scenario Analyst</h1>', unsafe_allow_html=True)
    st.markdown('<p class="mer-note">Describe a commercial scenario in plain English and see the '
                'financial impact. The AI extracts the assumptions; the deterministic engine does '
                'every calculation.</p>', unsafe_allow_html=True)

    with st.form("scenario_search", clear_on_submit=False):
        q = st.text_input("What can I model for you today?",
                          placeholder="What happens if prices increase 5%, volume falls 3% and OPEX increases by £1m?",
                          label_visibility="collapsed")
        run = st.form_submit_button("▶  Run Scenario", use_container_width=False)
    if run and q.strip():
        st.session_state.sc_chat = []
        _run_and_store(q.strip())

    # Example chips
    ex = st.columns(3)
    examples = ["Customer A volume +15%, prices +3%, raw material cost +5%, OPEX +£500k",
                "Market grows 5%, our volume +2%, prices +3%, costs +4%, OPEX +£750k",
                "We lose Customer C and Product D volume falls 10%"]
    for i, e in enumerate(examples):
        if ex[i].button(e, use_container_width=True):
            st.session_state.sc_chat = []
            _run_and_store(e)

    # Clarification
    if st.session_state.sc_clarify:
        st.warning("🤔 " + st.session_state.sc_clarify)

    plan = st.session_state.sc_plan
    result = st.session_state.sc_result

    # 1) SCENARIO UNDERSTOOD (editable)
    if plan and (plan.get("levers") or plan.get("opex_pct") or plan.get("opex_abs")):
        st.write("")
        panel_header("1 · Scenario understood", f"Interpreted by {st.session_state.sc_source} — review & edit, then re-run")
        rows = agent.assumptions_table(plan)
        st.markdown('<div class="mer-understood">', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame([{"Scope": "—", "Driver": "—", "Change": "—"}]),
                     hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("✏️ Edit assumptions"):
            _edit_assumptions(plan)

    # 2-5) Impact / drivers / bridges / commentary
    if result:
        st.write("")
        panel_header("2 · Scenario impact", "Current → scenario, all figures recomputed by the deterministic engine")
        _kpi_impact_cards(result)

        st.write("")
        c = st.columns([1.2, 1])
        with c[0]:
            panel_header("Current vs scenario", "Sales · Gross Margin · EBITDA")
            st.plotly_chart(charts.scenario_compare_bars(result["baseline"], result["scenario"]), use_container_width=True)
        with c[1]:
            panel_header("3 · EBITDA driver bridge", "Baseline → scenario")
            st.plotly_chart(charts.waterfall(scen_engine.scenario_ebitda_bridge(result)), use_container_width=True)

        st.write("")
        panel_header("4 · Bridges", "Sales, GM% and EBITDA — all reconcile exactly")
        bcols = st.columns(2)
        sb = scen_engine.scenario_sales_bridge(result)
        with bcols[0]:
            st.markdown("**Sales bridge**")
            st.plotly_chart(charts.waterfall(sb), use_container_width=True)
            st.caption(("✅ reconciles" if sb["reconciles"] else "⚠️") + f" · {fmt_m(sb['base'])} → {fmt_m(sb['current'])}")
        gb = scen_engine.scenario_gm_pct_bridge(result)
        with bcols[1]:
            st.markdown("**GM% bridge**")
            st.plotly_chart(charts.waterfall(gb, unit="pp"), use_container_width=True)
            st.caption("✅ reconciles" if gb["reconciles"] else "⚠️")

        st.write("")
        panel_header("5 · AI management commentary", "Explains only the deterministic scenario results")
        ans = st.session_state.sc_answer or ""
        st.markdown(f'<div class="mer-card mer-why">{ans}</div>', unsafe_allow_html=True)
        st.caption(f"Source: {st.session_state.sc_source} · grounded in calculated deltas")

        # 6) FOLLOW-UP CHAT
        st.write("")
        panel_header("6 · Follow-up", "Ask a follow-up — it modifies the scenario and re-runs the engine")
        for m in st.session_state.sc_chat:
            with st.chat_message(m["role"]):
                st.write(m["text"])
        fq = st.chat_input("Ask a follow-up question…  e.g. 'what if price is only 2%?'")
        if fq:
            st.session_state.sc_chat.append({"role": "user", "text": fq})
            # Re-parse the follow-up fresh (a new scenario relative to actuals)
            _run_and_store(fq)
            a = st.session_state.sc_answer or "(no change identified)"
            st.session_state.sc_chat.append({"role": "assistant", "text": a})
            st.rerun()

        # 7) SAVE + HISTORY
        st.write("")
        sv = st.columns([2, 1])
        name = sv[0].text_input("Scenario name", value=st.session_state.sc_understood or "Scenario", label_visibility="collapsed")
        if sv[1].button("💾 Save scenario", use_container_width=True):
            d = result["deltas"]
            st.session_state.sc_history.append({
                "Name": name, "Date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Assumptions": st.session_state.sc_understood,
                "Sales Δ": d["sales"]["delta"], "GM Δ": d["gross_margin"]["delta"],
                "EBITDA Δ": d["ebitda"]["delta"],
                "baseline": result["baseline"], "scenario": result["scenario"]})
            st.success(f"Saved “{name}”.")

    # SCENARIO HISTORY + COMPARISON
    if st.session_state.sc_history:
        st.write("")
        panel_header("Scenario history", "Saved scenarios — compare or delete")
        hist_view = pd.DataFrame([{"Name": h["Name"], "Date": h["Date"], "Assumptions": h["Assumptions"],
                                   "Sales Δ": fmt_signed_m(h["Sales Δ"]), "GM Δ": fmt_signed_m(h["GM Δ"]),
                                   "EBITDA Δ": fmt_signed_m(h["EBITDA Δ"])} for h in st.session_state.sc_history])
        st.dataframe(hist_view, hide_index=True, use_container_width=True)
        cc = st.columns([3, 1])
        picks = cc[0].multiselect("Compare scenarios", [h["Name"] for h in st.session_state.sc_history])
        if cc[1].button("🗑 Clear history"):
            st.session_state.sc_history = []
            st.rerun()
        if len(picks) >= 2:
            chosen = [h for h in st.session_state.sc_history if h["Name"] in picks]
            comp = {"Metric": ["Sales", "Gross Margin", "GM%", "EBITDA"]}
            for h in chosen:
                comp[h["Name"]] = [fmt_m(h["scenario"]["sales"]), fmt_m(h["scenario"]["gross_margin"]),
                                   f"{h['scenario']['gm_pct']*100:.1f}%", fmt_m(h["scenario"]["ebitda"])]
            panel_header("8 · Scenario comparison", "Side-by-side scenario outcomes")
            st.dataframe(pd.DataFrame(comp), hide_index=True, use_container_width=True)


def _edit_assumptions(plan):
    """Inline editor for the parsed levers; writes back to the plan and re-runs."""
    new_levers = []
    for i, lv in enumerate(plan.get("levers", [])):
        who = "Company-wide" if lv.get("scope") == "all" else lv.get("member", "?")
        st.markdown(f"**{who}**")
        cols = st.columns(4)
        vol = cols[0].number_input(f"Volume % [{i}]", value=float(lv.get("volume_pct", 0) * 100), step=1.0, key=f"ev{i}") / 100
        pr = cols[1].number_input(f"Price % [{i}]", value=float(lv.get("price_pct", 0) * 100), step=1.0, key=f"ep{i}") / 100
        cost = cols[2].number_input(f"Cost % [{i}]", value=float(lv.get("cost_pct", 0) * 100), step=1.0, key=f"ec{i}") / 100
        churn = cols[3].checkbox(f"Churn [{i}]", value=bool(lv.get("churn")), key=f"ech{i}")
        nlv = copy.deepcopy(lv); nlv.update({"volume_pct": vol, "price_pct": pr, "cost_pct": cost, "churn": churn})
        new_levers.append(nlv)
    oc = st.columns(2)
    opex_pct = oc[0].number_input("OPEX %", value=float(plan.get("opex_pct", 0) * 100), step=1.0, key="eopx") / 100
    opex_abs = oc[1].number_input("OPEX absolute (£)", value=float(plan.get("opex_abs", 0) or 0), step=50000.0, key="eopa")
    if st.button("🔁 Re-run with edited assumptions"):
        plan["levers"] = new_levers
        plan["opex_pct"] = opex_pct
        plan["opex_abs"] = opex_abs
        st.session_state.sc_plan = plan
        _rerun_plan()
        st.rerun()


# =========================================================================== #
def page_forecast():
    st.markdown('<h1>Forecast</h1>', unsafe_allow_html=True)
    c1, _ = st.columns([1, 3])
    horizon = c1.selectbox("Horizon (months)", [3, 6, 9, 12], index=1)
    bundle = fc_engine.scenario_bundle(DF, horizon); base_fc = bundle["Base"]
    if not base_fc.get("ok"):
        st.warning(base_fc.get("reason", "Cannot forecast.")); return
    k = st.columns(3)
    k[0].markdown(kpi_card("Base forecast (next period)", fmt_m(base_fc["values"][0]), tone="flat"), unsafe_allow_html=True)
    k[1].markdown(kpi_card(f"Base forecast ({horizon}m total)", fmt_m(base_fc["forecast_total"]), tone="flat"), unsafe_allow_html=True)
    k[2].markdown(kpi_card("Blended monthly growth", fmt_pct(base_fc["blended_monthly_growth"]), tone=_tone(base_fc["blended_monthly_growth"])), unsafe_allow_html=True)
    st.write(""); st.write("")
    panel_header("Scenario forecast", "Base / Upside / Downside — driver-based, explainable")
    st.plotly_chart(charts.scenario_chart(bundle), use_container_width=True)
    st.write("")
    panel_header("Why is this the forecast?", "Additive driver contributions (monthly)")
    st.dataframe(base_fc["explanation"], hide_index=True, use_container_width=True)


# =========================================================================== #
def page_budget():
    st.markdown('<h1>Budget</h1>', unsafe_allow_html=True)
    st.markdown('<p class="mer-note">Predictive next-year budget, built bottom-up from the last 12 '
                'months. Deterministic — adjust the assumptions and it recalculates instantly.</p>', unsafe_allow_html=True)
    c = st.columns(3)
    growth = c[0].slider("Revenue growth", -0.10, 0.30, 0.06, 0.005, format="%.1f%%")
    gm = c[1].slider("Target gross margin", 0.10, 0.85, float(round(calc.gm_pct(DF), 2)) if not np.isnan(calc.gm_pct(DF)) else 0.35, 0.005, format="%.0f%%")
    infl = c[2].slider("Cost inflation on COGS", 0.0, 0.15, 0.03, 0.005, format="%.1f%%")
    b = budget_engine.build_budget(DF, growth, gm, infl)
    if not b.get("ok"):
        st.warning(b.get("reason", "Cannot build budget.")); return
    ny = b["next_year"]; k = st.columns(3)
    k[0].markdown(kpi_card(f"{ny} budgeted revenue", fmt_m(b["budget_rev"]), delta=fmt_pct(growth) + " vs last 12m", tone=_tone(growth)), unsafe_allow_html=True)
    k[1].markdown(kpi_card(f"{ny} gross margin", f"{gm*100:.1f}%", delta=fmt_m(b["budget_gm_pounds"]) + " GM£", tone="flat"), unsafe_allow_html=True)
    k[2].markdown(kpi_card(f"{ny} budgeted COGS", fmt_m(b["budget_cogs"]), delta=fmt_pct(infl) + " cost inflation", tone="flat"), unsafe_allow_html=True)
    st.write(""); st.write("")
    panel_header(f"{ny} budget — quarterly phasing", "Seasonalised from the last four quarters")
    st.plotly_chart(charts.budget_phasing(b["q_budget"], b["last_year_actual_quarters"]), use_container_width=True)


# =========================================================================== #
def page_projection():
    st.markdown('<h1>5-Year Projection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="mer-note">Long-range revenue, gross margin and EBITDA — projected '
                'deterministically from the latest 12 months using explicit, editable assumptions.</p>', unsafe_allow_html=True)
    b = proj_engine.base_year(DF)
    if not b.get("ok"):
        st.warning(b.get("reason", "Cannot project.")); return
    seed = proj_engine.suggested_assumptions(DF); c = st.columns(4)
    years = c[0].selectbox("Horizon (years)", [3, 5, 7, 10], index=1)
    cagr = c[1].slider("Revenue growth (CAGR)", -0.15, 0.35, float(seed["revenue_cagr"]), 0.005, format="%.1f%%")
    gm_drift = c[2].slider("Annual GM% change (pp)", -0.03, 0.03, 0.0, 0.0025, format="%.2f")
    opex_eff = c[3].slider("Annual OPEX efficiency (pp of rev)", 0.0, 0.02, 0.0, 0.0025, format="%.2f")
    res = proj_engine.project(DF, years, cagr, gm_drift, opex_eff)
    bundle = proj_engine.scenario_bundle(DF, years, cagr, gm_drift, opex_eff); proj = res["projection"]
    st.write(""); k = st.columns(4)
    k[0].markdown(kpi_card(f"{proj['year'].iloc[-1]} revenue", fmt_m(res["exit_revenue"]), delta=fmt_pct(cagr) + " CAGR", tone=_tone(cagr)), unsafe_allow_html=True)
    k[1].markdown(kpi_card(f"{years}-yr cumulative revenue", fmt_m(res["cumulative_revenue"]), tone="flat"), unsafe_allow_html=True)
    k[2].markdown(kpi_card(f"{proj['year'].iloc[-1]} GM%", f"{res['exit_gm_pct']*100:.1f}%", delta=fmt_pp(gm_drift * years) + " vs base", tone=_tone(gm_drift)), unsafe_allow_html=True)
    k[3].markdown(kpi_card(f"{proj['year'].iloc[-1]} EBITDA", fmt_m(res["exit_ebitda"]), delta=fmt_m(res["cumulative_ebitda"]) + f" cum. {years}-yr", tone="flat"), unsafe_allow_html=True)
    st.write(""); st.write("")
    panel_header("Revenue & EBITDA projection", f"Base case · from {b['base_year']} actuals of {fmt_m(b['revenue'])}")
    st.plotly_chart(charts.projection_chart(proj), use_container_width=True)
    st.write("")
    panel_header("Scenario range", "Base / Upside / Downside revenue paths")
    st.plotly_chart(charts.projection_scenarios(bundle), use_container_width=True)


# =========================================================================== #
def page_bridges():
    st.markdown('<h1>Bridges</h1>', unsafe_allow_html=True)
    st.markdown('<p class="mer-note">Every bridge reconciles: base + drivers = current. '
                'Unattributable movement is shown explicitly as <b>Other / Residual</b>.</p>', unsafe_allow_html=True)
    panel_header("Sales bridge", f"{'Budget' if BASE=='budget' else 'Prior year'} → Actual")
    sb = bridges.sales_bridge(DF, BASE); st.plotly_chart(charts.waterfall(sb), use_container_width=True)
    st.caption(("✅ reconciles" if sb["reconciles"] else "⚠️ residual only") + f" · {fmt_m(sb['base'])} + drivers = {fmt_m(sb['current'])}")
    st.write("")
    panel_header("GM% bridge", "Ratio decomposition — percentage points (weighted methodology)")
    st.plotly_chart(charts.waterfall(bridges.gm_pct_bridge(DF, BASE), unit="pp"), use_container_width=True)
    st.write("")
    panel_header("EBITDA bridge", "Sales · Gross margin · OPEX effects")
    eb = bridges.ebitda_bridge(DF, BASE); st.plotly_chart(charts.waterfall(eb), use_container_width=True)
    st.caption(("✅ reconciles" if eb["reconciles"] else "⚠️ residual only") + f" · {fmt_m(eb['base'])} + drivers = {fmt_m(eb['current'])}")


# =========================================================================== #
def page_ai():
    st.markdown('<h1>AI Analyst</h1>', unsafe_allow_html=True)
    st.markdown('<p class="mer-note">Ask commercial questions about actuals. Answers are grounded in the '
                'calculation engine — the AI interprets calculated figures and never invents numbers.</p>', unsafe_allow_html=True)
    if not provider.available():
        st.info("No LLM configured — answers use the deterministic rule-based analyst (still fully grounded).")
    examples = ["Why is sales below budget?", "Why did GM% move?", "Which customers are causing the shortfall?", "Is the company beating the market?"]
    cols = st.columns(len(examples)); picked = None
    for i, ex in enumerate(examples):
        if cols[i].button(ex, use_container_width=True):
            picked = ex
    q = st.chat_input("Ask about sales, margin, drivers, market…")
    question = q or picked
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            res = analyst.answer_question(DF, question, BASE)
            st.write(res["answer"])
            st.caption(f"Source: {res['source']} · grounded in calculated facts")


# =========================================================================== #
def page_data():
    st.markdown('<h1>Data &amp; Quality</h1>', unsafe_allow_html=True)
    st.markdown('<p class="mer-note">Upload Excel/CSV — no rigid template. The system infers your '
                'columns, standardises them, and runs data-quality checks before analysis.</p>', unsafe_allow_html=True)
    up = st.file_uploader("Upload sales & margin data", type=["xlsx", "xls", "csv"])
    if st.button("↺ Load the built-in chemical demo dataset"):
        st.session_state.canonical = build_canonical(); st.session_state.source_label = "Demo — chemical manufacturing"
        for key in ["sc_plan", "sc_result", "sc_understood", "sc_answer", "sc_clarify"]:
            st.session_state[key] = None
        st.success("Loaded the demo dataset. Open the Executive Dashboard →")
    if up is not None:
        sheets = pipeline.read_upload(up); sheet = st.selectbox("Sheet", list(sheets.keys())); raw = sheets[sheet]
        st.markdown("**Preview of your upload**"); st.dataframe(raw.head(6), use_container_width=True)
        result = pipeline.process(raw, filename=up.name)
        st.write(""); panel_header("Data understanding", "Inferred semantic mapping (override not required)")
        st.dataframe(result["dictionary"], hide_index=True, use_container_width=True)
        for note in result["derivations"]:
            st.caption("• " + note)
        st.write(""); panel_header("Data quality", "Nothing important is silently ignored")
        for f in result["findings"]:
            icon = {"ok": "✓", "warn": "⚠", "error": "✗"}[f["status"]]
            (st.success if f["status"] == "ok" else st.warning if f["status"] == "warn" else st.error)(f"{icon} {f['message']}")
        if st.button("✅ Use this dataset for analysis"):
            st.session_state.canonical = result["canonical"]; st.session_state.source_label = f"Uploaded — {up.name}"
            for key in ["sc_plan", "sc_result", "sc_understood", "sc_answer", "sc_clarify"]:
                st.session_state[key] = None
            st.success(f"Loaded {len(result['canonical'])} rows. Open the Executive Dashboard →")


{"Executive Dashboard": page_dashboard, "Sales & Drivers": page_sales, "AI Scenario Analyst": page_scenarios,
 "Forecast": page_forecast, "Budget": page_budget, "5-Year Projection": page_projection,
 "Bridges": page_bridges, "AI Analyst": page_ai, "Data & Quality": page_data}[page]()
