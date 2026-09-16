"""AI Analyst — grounded commentary & Q&A. The AI only interprets calculated facts."""
from __future__ import annotations
import json
import numpy as np

from analytics import calculation_engine as calc
from analytics import bridges, attribution
from .llm_provider import get_provider


def _fmt_m(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return ("-" if v < 0 else "") + f"£{abs(v)/1e6:.1f}m"


def build_facts(df, base="budget"):
    k = calc.kpi_block(df)
    attr = attribution.attribute_sales(df, base)
    cust = bridges.driver_ranking(df, "customer", base)
    prod = bridges.driver_ranking(df, "product", base)
    market = None
    if "market_growth" in df.columns and df["market_growth"].notna().any():
        yoy = k["sales"]["py_pct"]
        market = {"market_growth": float(df["market_growth"].mean()),
                  "company_growth": None if yoy is None or (isinstance(yoy, float) and np.isnan(yoy)) else float(yoy)}
    return {"kpis": {"sales": k["sales"], "gross_margin": k["gross_margin"], "gm_pct": k["gm_pct"], "ebitda": k["ebitda"]},
            "sales_bridge": bridges.sales_bridge(df, base), "gm_pct_bridge": bridges.gm_pct_bridge(df, base),
            "ebitda_bridge": bridges.ebitda_bridge(df, base),
            "attribution": attr.get("effects", {}) if attr.get("ok") else {},
            "top_customers": cust, "top_products": prod, "market": market}


def rule_based_commentary(facts):
    s = facts["kpis"]["sales"]; gm = facts["kpis"]["gm_pct"]
    var, var_pct = s.get("variance"), s.get("variance_pct")
    if var is None or (isinstance(var, float) and np.isnan(var)):
        what = f"Sales were {_fmt_m(s['value'])}."
    else:
        what = f"Sales of {_fmt_m(s['value'])} were {_fmt_m(abs(var))} {'ahead of' if var >= 0 else 'below'} budget ({var_pct*100:+.1f}%)."
    eff = {k: v for k, v in facts.get("attribution", {}).items() if k != "Other / Residual"}
    why = ""
    if eff:
        top = sorted(eff.items(), key=lambda x: abs(x[1]), reverse=True)[0]
        why = f"The largest driver was {top[0].lower()} ({_fmt_m(top[1])})."
        neg, pos = facts["top_customers"]["negative"], facts["top_customers"]["positive"]
        if var is not None and var < 0 and neg:
            why += f" The shortfall is concentrated in {neg[0][0]} ({_fmt_m(neg[0][1])})."
        elif var is not None and var >= 0 and pos:
            why += f" Growth is led by {pos[0][0]} ({_fmt_m(pos[0][1])})."
    gm_move = gm.get("variance")
    if gm_move is not None and not (isinstance(gm_move, float) and np.isnan(gm_move)):
        why += f" GM% moved {gm_move*100:+.1f}pp vs budget."
    mkt = facts.get("market")
    if mkt and mkt.get("company_growth") is not None:
        cg, mg = mkt["company_growth"], mkt["market_growth"]
        why += f" The company grew {cg*100:+.1f}% versus a market up {mg*100:+.1f}% — a market-share {'loss' if cg < mg else 'gain'}."
    so_what = ""
    if var is not None and not (isinstance(var, float) and np.isnan(var)):
        so_what = f"If the current trend persists, full-year sales are likely to finish materially {'above' if var >= 0 else 'below'} plan; management should focus on the largest adverse driver above."
    return {"what": what, "why": why.strip(), "so_what": so_what, "source": "rule-based"}


def commentary(df, base="budget"):
    facts = build_facts(df, base); rb = rule_based_commentary(facts); provider = get_provider()
    if not provider.available():
        return {**rb, "facts": facts}
    system = ("You are a commercial finance director. You are given ONLY calculated figures in JSON, "
              "including a sales attribution with Volume/Price/Mix/New/Lost effects. You MUST NOT invent "
              "any number — quote figures exactly and reference the specific effects. Three labelled parts: "
              "WHAT, WHY (cite biggest effects and members), SO WHAT. 4-6 sentences.")
    payload = {"what_draft": rb["what"], "why_draft": rb["why"], "so_what_draft": rb["so_what"],
               "kpis": _slim(facts["kpis"]), "attribution": facts["attribution"], "top_customers": facts["top_customers"], "market": facts["market"]}
    try:
        return {"narrative": provider.complete(system, json.dumps(payload, default=_js)), "source": provider.name, "facts": facts, **rb}
    except Exception as e:
        return {**rb, "facts": facts, "llm_error": str(e)}


def answer_question(df, question, base="budget"):
    facts = build_facts(df, base); provider = get_provider()
    if not provider.available():
        rb = rule_based_commentary(facts)
        return {"answer": f"{rb['what']} {rb['why']}", "source": "rule-based", "facts": facts}
    system = ("You are an FP&A analyst. Answer using ONLY the JSON facts (pre-calculated, including a sales "
              "attribution). Never invent numbers. Quote figures exactly, cite specific effects/members, "
              "end with 'Management focus:' when relevant. 2-4 sentences.")
    try:
        return {"answer": provider.complete(system, json.dumps({"question": question, "facts": _slim_facts(facts)}, default=_js)), "source": provider.name, "facts": facts}
    except Exception as e:
        rb = rule_based_commentary(facts)
        return {"answer": f"{rb['what']} {rb['why']}", "source": "rule-based", "facts": facts, "llm_error": str(e)}


def _js(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return str(o)

def _slim(kpis):
    return {k: {kk: (None if isinstance(vv, float) and np.isnan(vv) else vv) for kk, vv in v.items() if kk in ("value", "budget", "variance", "variance_pct", "py_pct")} for k, v in kpis.items()}

def _slim_facts(facts):
    return {"kpis": _slim(facts["kpis"]), "attribution": facts["attribution"], "sales_bridge": facts["sales_bridge"]["steps"],
            "gm_pct_bridge": facts["gm_pct_bridge"]["steps"], "ebitda_bridge": facts["ebitda_bridge"]["steps"],
            "top_customers": facts["top_customers"], "top_products": facts["top_products"], "market": facts["market"]}
