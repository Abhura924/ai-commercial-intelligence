"""
AI Agent Router (v0.5)
======================
One entry point that decides what the user wants and dispatches to the right
capability — either a WHAT-IF SCENARIO (analytics/scenarios via scenario_agent)
or a DASHBOARD / BOARD PACK (ai/dashboard_builder). It can also answer a plain
analytical QUESTION (ai/analyst).

The LLM (or a rule-based fallback) only CLASSIFIES intent and extracts parameters —
every number still comes from the deterministic engine.

Intents:
    scenario  : a hypothetical change to model ("what if …", "grow X 10%", "+5% price")
    dashboard : build/show a view or board pack ("build me a dashboard", "show KPIs …")
    question  : ask about the actuals ("why is sales below budget?")
"""
from __future__ import annotations
import json
import re

from .llm_provider import get_provider
from . import scenario_agent as sa
from . import dashboard_builder as db

# --- rule-based signals ------------------------------------------------------ #
_SCENARIO_SIGNALS = ["what if", "what happens if", "if we", "if our", "if customer", "if product",
                     "increase", "decrease", "grow", "shrink", "raise", "cut", "reduce", "boost",
                     "uplift", "reprice", "churn", "lose ", "lost ", "simulate", "model ", "assume",
                     "scenario", "+", "-", "%", "£", "$", "drop ", "fall", "rise", "rises", "growth of"]
_DASHBOARD_SIGNALS = ["dashboard", "board", "board pack", "deck", "slide", "build me", "create me",
                      "show me", "show the", "present", "put together", "assemble", "overview",
                      "summary view", "board meeting", "meeting in", "kpi", "kpis", "visual", "report",
                      "pack for", "view with", "give me a"]
_QUESTION_SIGNALS = ["why", "which", "who", "where", "how much", "how many", "is the", "are we",
                     "did we", "what is", "what are", "explain", "tell me about", "what drove"]


def _rule_based_intent(text):
    t = text.lower()
    dash = sum(1 for s in _DASHBOARD_SIGNALS if s in t)
    scen = sum(1 for s in _SCENARIO_SIGNALS if s in t)
    ques = sum(1 for s in _QUESTION_SIGNALS if s in t)
    # A dashboard ask usually names blocks/views AND has no hypothetical change verb.
    has_change = any(w in t for w in ["what if", "what happens if", "if we", "if our", "increase",
                                      "decrease", "grow", "raise", "cut", "reduce", "churn", "lose",
                                      "reprice", "boost", "uplift", "+", "%", "£", "$"])
    # Strong dashboard phrasing wins even if a % sneaks in (e.g. "show sales vs budget %").
    strong_dash = any(s in t for s in ["dashboard", "board pack", "board meeting", "build me", "create me a dashboard", "deck", "assemble"])
    if strong_dash and not ("what if" in t or "what happens if" in t):
        return "dashboard"
    if has_change and scen >= dash:
        return "scenario"
    if dash > 0 and dash >= scen:
        return "dashboard"
    if ques > 0 and not has_change:
        return "question"
    # default: if it mentions a numeric change -> scenario, else dashboard
    return "scenario" if has_change else "dashboard"


def classify_intent(text):
    """Return (intent, source). intent in {scenario, dashboard, question}."""
    provider = get_provider()
    if provider.available():
        try:
            system = ("Classify a finance user's request into exactly one intent and return ONLY JSON "
                      '{"intent":"scenario|dashboard|question"}. '
                      "scenario = a hypothetical change to model (what-if, +/- a driver, grow/lose a "
                      "customer, price/cost/volume/OPEX change). "
                      "dashboard = build or show a view / board pack / KPIs / charts (no hypothetical change). "
                      "question = ask about the existing actuals (why/which/who).")
            raw = provider.complete(system, text, max_tokens=60)
            m = re.search(r'"intent"\s*:\s*"(scenario|dashboard|question)"', raw)
            if m:
                return m.group(1), provider.name
        except Exception:
            pass
    return _rule_based_intent(text), "rule-based"


def route(df, text, base="budget"):
    """
    Full agent turn. Returns a dict with:
        kind: "scenario" | "dashboard" | "question"
        ... plus kind-specific payload.
    For scenario, if the parser can't find a concrete change we fall back to a
    dashboard or question so the user still gets something useful.
    """
    intent, src = classify_intent(text)

    if intent == "dashboard":
        spec, dsrc = db.build_spec(text)
        return {"kind": "dashboard", "intent_source": src, "spec": spec, "spec_source": dsrc, "request": text}

    if intent == "question":
        from . import analyst
        res = analyst.answer_question(df, text, base)
        return {"kind": "question", "intent_source": src, "answer": res["answer"],
                "answer_source": res["source"], "facts": res.get("facts"), "request": text}

    # scenario (default)
    ask = sa.ask(df, text, base)
    if ask.get("clarification"):
        return {"kind": "scenario", "intent_source": src, "clarification": ask["clarification"],
                "plan": ask.get("plan"), "request": text}
    if ask.get("result") is None:
        # Couldn't parse a concrete change. If it *looks* like a view, build one.
        if any(s in text.lower() for s in _DASHBOARD_SIGNALS):
            spec, dsrc = db.build_spec(text)
            return {"kind": "dashboard", "intent_source": src, "spec": spec, "spec_source": dsrc, "request": text}
        # otherwise answer as a question
        from . import analyst
        res = analyst.answer_question(df, text, base)
        return {"kind": "question", "intent_source": src, "answer": res["answer"],
                "answer_source": res["source"], "facts": res.get("facts"), "request": text}
    return {"kind": "scenario", "intent_source": src, "plan": ask["plan"], "result": ask["result"],
            "understood": ask["understood"], "answer": ask["answer"], "answer_source": ask["source"],
            "scope_label": ask.get("scope_label", "current actuals"), "request": text}
