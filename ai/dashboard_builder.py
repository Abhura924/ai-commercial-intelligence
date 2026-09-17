"""
Natural-Language Dashboard Builder (v0.4)
=========================================
"I have a board meeting in 15 minutes — build me a dashboard with the 5 key KPIs,
sales vs budget, GM%, EBITDA, a sales bridge and the top 3 issues."

The request is turned into a DASHBOARD SPEC (which blocks to render) — the LLM (or
a rule-based fallback) only chooses WHICH blocks are relevant. Every number in
those blocks still comes from the deterministic engine. No arithmetic by the LLM.
"""
from __future__ import annotations
import json
import re
from .llm_provider import get_provider

# Catalogue of available blocks the builder can assemble.
BLOCKS = {
    "kpis":         "Executive KPI cards (Sales, GM, GM%, EBITDA, Sales vs Budget)",
    "sales_trend":  "Sales performance trend (actual vs budget vs prior year)",
    "gm_trend":     "GM% trend",
    "sales_bridge": "Sales variance bridge (budget → actual)",
    "gm_bridge":    "GM% variance bridge",
    "ebitda_bridge":"EBITDA variance bridge",
    "drivers":      "Top commercial drivers by customer & product",
    "risks_opps":   "Risks & opportunities",
    "commentary":   "AI management commentary (What / Why / So what)",
    "forecast":     "Forward forecast (Base/Upside/Downside)",
}
DEFAULT_SPEC = ["kpis", "sales_trend", "sales_bridge", "drivers", "commentary", "risks_opps"]

_KEYWORDS = {
    "kpis":         ["kpi", "kpis", "key metric", "headline", "five most important", "5 most important", "scorecard", "sales vs budget", "ebitda", "gm%", "gross margin"],
    "sales_trend":  ["trend", "over time", "monthly", "performance", "sales performance"],
    "gm_trend":     ["gm% trend", "margin trend", "gross margin trend"],
    "sales_bridge": ["sales bridge", "bridge", "variance bridge", "waterfall", "drivers of sales"],
    "gm_bridge":    ["gm bridge", "margin bridge", "gm% bridge"],
    "ebitda_bridge":["ebitda bridge"],
    "drivers":      ["driver", "biggest", "top customer", "top product", "contributor", "who drove"],
    "risks_opps":   ["risk", "opportunit", "issues", "biggest issues", "concerns", "watch"],
    "commentary":   ["commentary", "summary", "explain", "narrative", "what happened", "so what"],
    "forecast":     ["forecast", "outlook", "next", "projection", "what happens next", "coming"],
}


def rule_based_spec(text):
    t = text.lower()
    chosen = [b for b, kws in _KEYWORDS.items() if any(k in t for k in kws)]
    # de-dup preserving catalogue order
    chosen = [b for b in BLOCKS if b in chosen]
    if not chosen:
        return list(DEFAULT_SPEC)
    # KPIs and commentary are almost always wanted for a board view
    if "kpis" not in chosen:
        chosen = ["kpis"] + chosen
    if "commentary" not in chosen and ("board" in t or "management" in t or "executive" in t):
        chosen.append("commentary")
    return chosen


def build_spec(text):
    """Return (ordered list of block ids, source)."""
    provider = get_provider()
    if provider.available():
        try:
            system = ("You assemble a board dashboard. Choose which blocks to show from this catalogue "
                      "and return ONLY a JSON list of block ids in display order. Catalogue: "
                      + json.dumps(BLOCKS) + ". Always include 'kpis'. Do not invent ids.")
            raw = provider.complete(system, text, max_tokens=200)
            ids = _parse_list(raw)
            ids = [b for b in ids if b in BLOCKS]
            if ids:
                if "kpis" not in ids:
                    ids = ["kpis"] + ids
                return ids, provider.name
        except Exception:
            pass
    return rule_based_spec(text), "rule-based"


def _parse_list(text):
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        val = json.loads(cleaned)
        if isinstance(val, list):
            return [str(x) for x in val]
    except Exception:
        pass
    # fall back: pull known ids out of the text
    return [b for b in BLOCKS if b in cleaned]
