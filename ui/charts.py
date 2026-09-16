"""Professional Plotly chart builders (consistent executive styling)."""
import numpy as np
import plotly.graph_objects as go
from .theme import TEAL, BRICK, AMBER, INK, MUTED, LINE
from .theme import fmt_m, fmt_signed_m


def _layout(fig, title=None, height=340):
    fig.update_layout(
        title=dict(text=title or "", font=dict(family="Source Serif 4", size=16, color=INK)),
        height=height, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=44 if title else 14, b=28, l=12, r=14),
        font=dict(family="IBM Plex Sans", color=INK, size=12),
        legend=dict(orientation="h", y=-0.2, x=0, font=dict(size=12)))
    fig.update_xaxes(showgrid=False, linecolor=LINE, ticks="outside", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#EEECE3", zeroline=False, tickfont=dict(color=MUTED))
    return fig


def sales_trend(ts):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["period"], y=ts["sales"], name="Actual", mode="lines+markers", line=dict(color=TEAL, width=3), marker=dict(size=5)))
    if "budget" in ts and ts["budget"].notna().any():
        fig.add_trace(go.Scatter(x=ts["period"], y=ts["budget"], name="Budget", line=dict(color=AMBER, width=2, dash="dash")))
    if "prior_year" in ts and ts["prior_year"].notna().any():
        fig.add_trace(go.Scatter(x=ts["period"], y=ts["prior_year"], name="Prior year", line=dict(color="#9CA0A8", width=1.75)))
    fig.update_yaxes(tickprefix="£", tickformat=".2s")
    return _layout(fig)


def gm_trend(ts):
    fig = go.Figure()
    if "gm_pct" in ts:
        fig.add_trace(go.Scatter(x=ts["period"], y=ts["gm_pct"], name="Actual GM%", mode="lines+markers", line=dict(color=TEAL, width=3), marker=dict(size=5)))
    if "budget_gm_pct" in ts and ts["budget_gm_pct"].notna().any():
        fig.add_trace(go.Scatter(x=ts["period"], y=ts["budget_gm_pct"], name="Budget GM%", line=dict(color=AMBER, width=2, dash="dash")))
    if "py_gm_pct" in ts and ts["py_gm_pct"].notna().any():
        fig.add_trace(go.Scatter(x=ts["period"], y=ts["py_gm_pct"], name="Prior year GM%", line=dict(color="#9CA0A8", width=1.75)))
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig)


def waterfall(bridge, unit="m"):
    labels = [bridge["base_label"]] + [s[0] for s in bridge["steps"]] + ["Current"]
    measures = ["absolute"] + ["relative"] * len(bridge["steps"]) + ["total"]
    values = [bridge["base"]] + [s[1] for s in bridge["steps"]] + [bridge["current"]]
    if unit == "pp":
        text = [f"{bridge['base']*100:.1f}%"] + [f"{s[1]*100:+.2f}pp" for s in bridge["steps"]] + [f"{bridge['current']*100:.1f}%"]
    else:
        text = [fmt_m(bridge["base"])] + [fmt_signed_m(s[1]) for s in bridge["steps"]] + [fmt_m(bridge["current"])]
    fig = go.Figure(go.Waterfall(orientation="v", measure=measures, x=labels, y=values, text=text,
        textposition="outside", textfont=dict(size=11, color=INK), connector={"line": {"color": "#DEDCD2", "dash": "dot"}},
        increasing={"marker": {"color": TEAL}}, decreasing={"marker": {"color": BRICK}}, totals={"marker": {"color": INK}}))
    fig.update_yaxes(tickformat=".0%") if unit == "pp" else fig.update_yaxes(tickprefix="£", tickformat=".2s")
    return _layout(fig, height=380)


def scenario_chart(bundle):
    fig = go.Figure(); base = bundle["Base"]
    if base.get("ok"):
        h = base["history"]; fig.add_trace(go.Scatter(x=h["period"], y=h["sales"], name="Actual", mode="lines", line=dict(color=INK, width=2.5)))
    colors = {"Base": TEAL, "Upside": AMBER, "Downside": BRICK}
    for scen, fc in bundle.items():
        if fc.get("ok"):
            fig.add_trace(go.Scatter(x=fc["periods"], y=fc["values"], name=scen, mode="lines+markers", line=dict(color=colors[scen], width=2, dash="dash"), marker=dict(size=4)))
    fig.update_yaxes(tickprefix="£", tickformat=".2s")
    return _layout(fig, height=360)


def projection_chart(p):
    y = p["year"].astype(str); fig = go.Figure()
    fig.add_trace(go.Bar(x=y, y=p["revenue"], name="Revenue", marker_color=TEAL, text=[fmt_m(v) for v in p["revenue"]], textposition="outside"))
    fig.add_trace(go.Scatter(x=y, y=p["ebitda"], name="EBITDA", mode="lines+markers", line=dict(color=AMBER, width=2.5), marker=dict(size=6)))
    fig.update_yaxes(tickprefix="£", tickformat=".2s")
    return _layout(fig, height=360)


def projection_scenarios(bundle):
    colors = {"Base": TEAL, "Upside": AMBER, "Downside": BRICK}; fig = go.Figure()
    for scen, res in bundle.items():
        if res.get("ok"):
            p = res["projection"]; fig.add_trace(go.Scatter(x=p["year"].astype(str), y=p["revenue"], name=scen, mode="lines+markers", line=dict(color=colors[scen], width=2.5, dash="solid" if scen == "Base" else "dash"), marker=dict(size=5)))
    fig.update_yaxes(tickprefix="£", tickformat=".2s")
    return _layout(fig, height=360)


def budget_phasing(q_budget, last_year):
    labels = list(q_budget.keys()); vals = list(q_budget.values()); ly = list(last_year.values())
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=vals, name="Budget", marker_color=TEAL, text=[fmt_m(v) for v in vals], textposition="outside"))
    fig.add_trace(go.Scatter(x=labels, y=ly, name="Last year actual", mode="lines+markers", line=dict(color=MUTED, width=2, dash="dash")))
    fig.update_yaxes(tickprefix="£", tickformat=".2s")
    return _layout(fig, height=340)


def driver_bars(ranking):
    names = [c for c, _ in ranking["positive"]][::-1] + [c for c, _ in ranking["negative"]][::-1]
    vals = [v for _, v in ranking["positive"]][::-1] + [v for _, v in ranking["negative"]][::-1]
    colors = [TEAL if v >= 0 else BRICK for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors, text=[fmt_signed_m(v) for v in vals], textposition="outside"))
    fig.update_xaxes(tickprefix="£", tickformat=".2s")
    return _layout(fig, height=max(260, 40 * len(names)))


def effect_drill_bars(df, effect, by):
    if df is None or df.empty:
        return _layout(go.Figure(), height=200)
    d = df.sort_values(effect); colors = [TEAL if v >= 0 else BRICK for v in d[effect]]
    fig = go.Figure(go.Bar(x=d[effect], y=d[by], orientation="h", marker_color=colors, text=[fmt_signed_m(v) for v in d[effect]], textposition="outside"))
    fig.update_xaxes(tickprefix="£", tickformat=".2s")
    return _layout(fig, height=max(220, 38 * len(d)))


def scenario_compare_bars(baseline, scenario):
    metrics = ["sales", "gross_margin", "ebitda"]; labels = ["Sales", "Gross Margin", "EBITDA"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[baseline[m] for m in metrics], name="Current", marker_color=MUTED, text=[fmt_m(baseline[m]) for m in metrics], textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=[scenario[m] for m in metrics], name="Scenario", marker_color=TEAL, text=[fmt_m(scenario[m]) for m in metrics], textposition="outside"))
    fig.update_yaxes(tickprefix="£", tickformat=".2s"); fig.update_layout(barmode="group")
    return _layout(fig, height=340)
