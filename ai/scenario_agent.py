"""
AI Scenario Agent — natural language → structured levers → deterministic engine.
The LLM only extracts parameters; all arithmetic is in analytics/scenarios.py.
v0.4: also extracts a FORECAST HORIZON so a scenario can be applied over a future
period (via analytics/forecast.forecast_frame) instead of the trailing actuals.
"""
from __future__ import annotations
import json
import re
import difflib
import numpy as np

from analytics import scenarios as scen_engine
from .llm_provider import get_provider

SCENARIO_HINTS = ["what if", "increase", "decrease", "grow", "shrink", "raise", "cut", "drop",
                  "lose", "lost", "churn", "add", "reduce", "boost", "uplift", "reprice",
                  "+", "-", "%", "scenario", "simulate", "model", "assume", "forecast"]


def looks_like_scenario(text):
    t = text.lower()
    return any(h in t for h in SCENARIO_HINTS)


def _all_members(df):
    out = {}
    for dim in ["customer", "product", "region", "business_unit"]:
        if dim in df.columns:
            for m in df[dim].dropna().unique():
                out[str(m).lower()] = (dim, str(m))
    return out


def _resolve_member(name, idx):
    if not name: return None, None
    key = name.strip().lower()
    if key in idx: return idx[key]
    match = difflib.get_close_matches(key, list(idx.keys()), n=1, cutoff=0.6)
    if match: return idx[match[0]]
    for k, v in idx.items():
        if key in k or k in key: return v
    return None, None


_PCT = r"([+-]?\d+(?:\.\d+)?)\s*%"
_NEG_WORDS = ["cut", "reduce", "lower", "decrease", "shrink", "down", "fall", "drop by", "decline"]
_DEC_VERB = ["lose", "lost", "churn", "remove", "exit", "drop "]
_COST_WORDS = ("cost", "cogs", "input", "raw material", "raw-material", "raw")
_COMPANY_WORDS = ("we ", "our ", "company", "overall", "across", "group", "sales volume",
                  "total volume", "whole", "everything", "all customers", "all products",
                  "sales", "revenue", "turnover")
_MONEY = r"[£$]?\s*([\d,.]+)\s*(k|m|bn|thousand|million|billion)?\b"


def _parse_money(s):
    m = re.search(_MONEY, s)
    if not m: return None
    try: val = float(m.group(1).replace(",", ""))
    except ValueError: return None
    mult = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}.get((m.group(2) or "").lower(), 1.0)
    return val * mult


def _clauses(text):
    t = re.sub(r"[,;]+", "|", text.lower())
    t = re.sub(r"\s+(and|but|then|also|while|plus)\s+", "|", t)
    return [c.strip() for c in t.split("|") if c.strip()]


def _parse_clause(clause, idx):
    found = None
    for key, (dim, name) in sorted(idx.items(), key=lambda kv: -len(kv[0])):
        if key in clause: found = (dim, name); break
    if not found: return None
    dim, name = found
    lever = scen_engine.blank_lever(); lever.update({"dimension": dim, "member": name, "scope": "member"})
    if any(w in clause for w in _DEC_VERB): lever["churn"] = True; return lever
    pm = re.search(_PCT, clause)
    if not pm: return None
    pct = float(pm.group(1)) / 100
    neg = pm.group(1).strip().startswith("-") or any(w in clause for w in _NEG_WORDS)
    signed = -abs(pct) if neg else abs(pct)
    if "price" in clause or "asp" in clause: lever["price_pct"] = signed
    elif any(w in clause for w in _COST_WORDS): lever["cost_pct"] = signed
    else: lever["volume_pct"] = signed
    return lever


def extract_horizon(text):
    """Return horizon in months if a forecast period is mentioned, else None."""
    t = text.lower()
    period_word = re.search(r"\d+\s*(month|months|mo|quarter|quarters|q|year|years|yr)\b", t)
    trigger = any(w in t for w in ("forecast", "next", "over the", "coming", "future"))
    if not trigger and not period_word:
        return None
    m = re.search(r"(\d+)\s*(month|months|mo)\b", t)
    if m: return int(m.group(1))
    m = re.search(r"(\d+)\s*(quarter|quarters|q)\b", t)
    if m: return int(m.group(1)) * 3
    m = re.search(r"(\d+)\s*(year|years|yr)\b", t)
    if m: return int(m.group(1)) * 12
    # word-number periods without a digit
    if re.search(r"next\s+quarter|coming\s+quarter", t): return 3
    if re.search(r"next\s+year|coming\s+year", t): return 12
    if re.search(r"next\s+month|coming\s+month", t): return 1
    # generic 'forecast'/'next'/'coming' with no number -> default 6 months
    if trigger: return 6
    return None


def rule_based_parse(text, df):
    idx = _all_members(df)
    levers = []; opex_pct = 0.0; opex_abs = 0.0; market_growth = None; prev_member = None
    for clause in _clauses(text):
        is_opex = any(w in clause for w in ("opex", "overhead", "sg&a", "operating expense"))
        is_cost = any(w in clause for w in _COST_WORDS)
        neg = any(w in clause for w in _NEG_WORDS)
        if is_opex:
            if "£" in clause or "$" in clause or re.search(r"\d\s*(k|m|bn|thousand|million|billion)\b", clause):
                amt = _parse_money(clause)
                if amt is not None: opex_abs = -abs(amt) if neg else abs(amt); continue
            if re.search(_PCT, clause):
                v = float(re.search(_PCT, clause).group(1)) / 100
                opex_pct = -abs(v) if neg else abs(v); continue
        mkt = re.search(r"market(?:\s+\w+){0,3}?\s*" + _PCT, clause) or re.search(_PCT + r"\s*(?:\w+\s+){0,3}?market", clause)
        has_driver = any(w in clause for w in ("sales", "revenue", "turnover", "volume", "price", "asp", "qty", "units")) or is_cost
        if mkt and not has_driver:
            market_growth = float(mkt.group(1)) / 100; continue
        lever = _parse_clause(clause, idx)
        if lever is None:
            pm = re.search(_PCT, clause)
            if pm:
                pct = float(pm.group(1)) / 100; signed = -abs(pct) if neg else abs(pct)
                metric = "price_pct" if ("price" in clause or "asp" in clause) else ("cost_pct" if is_cost else "volume_pct")
                if is_cost or any(w in clause for w in _COMPANY_WORDS):
                    lv = scen_engine.blank_lever(); lv.update({"scope": "all", "member": "*", "dimension": "customer", metric: signed})
                    levers.append(lv); prev_member = None; continue
                if prev_member is not None: prev_member[metric] = signed
            continue
        existing = next((l for l in levers if l["member"] == lever["member"] and l["dimension"] == lever["dimension"]), None)
        if existing:
            for k in ["volume_pct", "price_pct", "cost_pct"]:
                if lever[k]: existing[k] = lever[k]
            existing["churn"] = existing["churn"] or lever["churn"]; prev_member = existing
        else:
            levers.append(lever); prev_member = lever
    return {"levers": levers, "opex_pct": opex_pct, "opex_abs": opex_abs,
            "market_growth": market_growth, "horizon": extract_horizon(text)}


def llm_parse(text, df, provider):
    dims = {d: list(map(str, df[d].dropna().unique()))[:40] for d in ["customer", "product", "region", "business_unit"] if d in df.columns}
    system = ("You convert a finance user's what-if question into STRUCTURED LEVERS. You do NOT do maths. "
              "Return ONLY strict JSON: {\"levers\":[{\"dimension\":\"customer|product|region|business_unit\","
              "\"member\":\"<exact name or *>\",\"volume_pct\":0.0,\"price_pct\":0.0,\"cost_pct\":0.0,\"churn\":false}],"
              "\"opex_pct\":0.0,\"opex_abs\":0.0,\"market_growth\":null,\"horizon_months\":null}. Percentages are "
              "fractions (10%->0.10; a cut is negative). opex_abs is absolute currency (+£500k->500000). "
              "horizon_months is set only if the user asks to apply it over a forecast period. Use member names EXACTLY as given.")
    raw = provider.complete(system, json.dumps({"question": text, "available_members": dims}), max_tokens=500)
    parsed = _parse_json(raw)
    if not parsed: return None
    idx = _all_members(df); out = []
    for lv in parsed.get("levers", []):
        base = scen_engine.blank_lever(); member = lv.get("member")
        if member in ("*", "all", None):
            base.update({"scope": "all", "member": "*", "dimension": lv.get("dimension", "customer")})
        else:
            dim, name = _resolve_member(member, idx)
            if name is None: dim, name = lv.get("dimension", "customer"), member
            base.update({"scope": "member", "dimension": dim, "member": name})
        for k in ["volume_pct", "price_pct", "cost_pct"]:
            try: base[k] = float(lv.get(k, 0.0) or 0.0)
            except (TypeError, ValueError): base[k] = 0.0
        base["churn"] = bool(lv.get("churn", False))
        if base["churn"] or base["volume_pct"] or base["price_pct"] or base["cost_pct"]: out.append(base)
    def _num(key):
        try: return float(parsed.get(key, 0.0) or 0.0)
        except (TypeError, ValueError): return 0.0
    hz = parsed.get("horizon_months")
    try: hz = int(hz) if hz else extract_horizon(text)
    except (TypeError, ValueError): hz = extract_horizon(text)
    return {"levers": out, "opex_pct": _num("opex_pct"), "opex_abs": _num("opex_abs"),
            "market_growth": parsed.get("market_growth"), "horizon": hz}


def _parse_json(text):
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try: return json.loads(cleaned)
    except Exception:
        try: return json.loads(cleaned[cleaned.index("{"):cleaned.rindex("}") + 1])
        except Exception: return None


def _fmt_m(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "n/a"
    a = abs(v); sign = "-" if v < 0 else ""
    if a >= 1e6: return f"{sign}£{a/1e6:.1f}m"
    if a >= 1e3: return f"{sign}£{a/1e3:.0f}k"
    return f"{sign}£{a:,.0f}"


def _pctlabel(v):
    pct = v * 100
    return f"{pct:+.1f}%" if abs(pct) < 1 else f"{pct:+.0f}%"


def _describe_levers(plan):
    parts = []
    for lv in plan["levers"]:
        who = "Company-wide" if lv.get("scope") == "all" else lv.get("member")
        bits = []
        if lv.get("churn"): bits.append("removed")
        if lv.get("volume_pct"): bits.append(f"volume {_pctlabel(lv['volume_pct'])}")
        if lv.get("price_pct"): bits.append(f"price {_pctlabel(lv['price_pct'])}")
        if lv.get("cost_pct"): bits.append(f"cost {_pctlabel(lv['cost_pct'])}")
        if bits: parts.append(f"{who} ({', '.join(bits)})")
    if plan.get("opex_pct"): parts.append(f"OPEX {_pctlabel(plan['opex_pct'])}")
    if plan.get("opex_abs"): parts.append(f"OPEX {_fmt_m(plan['opex_abs'])}")
    if plan.get("market_growth") is not None: parts.append(f"Market growth {_pctlabel(plan['market_growth'])} (context)")
    return "; ".join(parts) if parts else None


def assumptions_table(plan):
    rows = []
    for lv in plan.get("levers", []):
        who = "Company-wide" if lv.get("scope") == "all" else lv.get("member")
        if lv.get("churn"): rows.append({"Scope": who, "Driver": "Churn (remove)", "Change": "removed"})
        for metric, label in [("volume_pct", "Volume"), ("price_pct", "Price"), ("cost_pct", "Cost")]:
            if lv.get(metric): rows.append({"Scope": who, "Driver": label, "Change": f"{lv[metric]*100:+.1f}%"})
    if plan.get("opex_pct"): rows.append({"Scope": "Company-wide", "Driver": "OPEX", "Change": f"{plan['opex_pct']*100:+.1f}%"})
    if plan.get("opex_abs"): rows.append({"Scope": "Company-wide", "Driver": "OPEX", "Change": _fmt_m(plan["opex_abs"])})
    if plan.get("market_growth") is not None: rows.append({"Scope": "Market", "Driver": "Market growth", "Change": f"{plan['market_growth']*100:+.1f}%"})
    if plan.get("horizon"): rows.append({"Scope": "Applied over", "Driver": "Forecast horizon", "Change": f"{plan['horizon']} months"})
    return rows


def needs_clarification(text, plan):
    t = text.lower()
    has_any = bool(plan.get("levers") or plan.get("opex_pct") or plan.get("opex_abs"))
    mentions = re.search(r"(price|volume|asp|sales)", t) and re.search(_PCT, t)
    company = any(w in t for w in ["company", "overall", "across the board", "whole", "all customers", "all products", "everything", "we ", "our ", "sales", "revenue", "turnover"])
    if mentions and not has_any and not company:
        return ("Should that change apply company-wide, or to a specific customer or product? "
                "For example: “increase price 5% for Customer A” or “company-wide price +5%”.")
    return None


def parse_only(df, question):
    provider = get_provider(); plan, source = None, "rule-based"
    if provider.available():
        try: plan = llm_parse(question, df, provider); source = provider.name
        except Exception: plan = None
    if not plan or (not plan.get("levers") and not plan.get("opex_pct") and not plan.get("opex_abs")):
        plan = rule_based_parse(question, df); source = "rule-based" if not provider.available() else source
    return plan, source


def _base_frame(df, plan):
    """Return (base_df, scope_label) — actuals, or a forecast frame if horizon set."""
    hz = plan.get("horizon")
    if hz:
        from analytics import forecast as fc
        ff = fc.forecast_frame(df, horizon=hz, scenario="Base")
        if ff.get("ok"):
            label = f"{hz}-month forecast ({ff['period_start'].strftime('%b %Y')}–{ff['period_end'].strftime('%b %Y')})"
            return ff["frame"], label
    return df, "current actuals"


def run_plan(df, plan, base="budget"):
    base_df, scope_label = _base_frame(df, plan)
    result = scen_engine.run_scenario(base_df, plan.get("levers", []), plan.get("opex_pct", 0.0), plan.get("opex_abs", 0.0))
    result["scope_label"] = scope_label
    understood = _describe_levers(plan); d = result["deltas"]
    rb = (f"Applying {understood or 'no changes'} over {scope_label} changes Sales by {_fmt_m(d['sales']['delta'])}, "
          f"Gross Margin by {_fmt_m(d['gross_margin']['delta'])}, and EBITDA by {_fmt_m(d['ebitda']['delta'])} "
          f"(GM% {d['gm_pct']['delta']*100:+.2f}pp) versus the {scope_label} baseline.")
    answer, source = rb, "rule-based"
    provider = get_provider()
    if provider.available():
        try:
            system = ("You are an FP&A analyst narrating a what-if in three short labelled parts: WHAT HAPPENED, "
                      "WHY, SO WHAT. Given ONLY recalculated deltas and driver bridge steps. Never invent numbers; "
                      "quote exactly. 4-6 sentences.")
            payload = {"scenario": understood, "applied_over": scope_label,
                       "deltas": {k: {kk: (None if isinstance(vv, float) and np.isnan(vv) else vv) for kk, vv in v.items()} for k, v in d.items()},
                       "sales_bridge": scen_engine.scenario_sales_bridge(result)["steps"],
                       "ebitda_bridge": scen_engine.scenario_ebitda_bridge(result)["steps"]}
            answer = provider.complete(system, json.dumps(payload, default=_js)); source = provider.name
        except Exception: answer = rb
    return {"result": result, "understood": understood, "answer": answer, "source": source, "scope_label": scope_label}


def ask(df, question, base="budget"):
    plan, source = parse_only(df, question)
    clar = needs_clarification(question, plan)
    if clar:
        return {"plan": plan, "result": None, "understood": _describe_levers(plan), "source": source, "clarification": clar, "answer": clar}
    understood = _describe_levers(plan)
    if not understood:
        return {"plan": plan, "result": None, "understood": None, "source": source, "clarification": None,
                "answer": ("I couldn't identify a concrete scenario. Try e.g. \u201cgrow Customer A by 10%\u201d, "
                           "\u201craise Product B price 5%\u201d, \u201close Customer C\u201d, \u201cOPEX up £500k\u201d, "
                           "or add a horizon like \u201cover the next 6 months\u201d.")}
    run = run_plan(df, plan, base)
    return {"plan": plan, "result": run["result"], "understood": understood, "source": run["source"],
            "answer": run["answer"], "clarification": None, "scope_label": run["scope_label"]}


def _js(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return str(o)
