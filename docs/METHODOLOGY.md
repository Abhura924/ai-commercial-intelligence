# Calculation, Bridge, Attribution & Scenario Methodology

All figures come from analytics/* (deterministic). The LLM never does arithmetic —
in the Scenario Analyst it only extracts assumptions from natural language.

## Core
Sales=Σrevenue · COGS=Σcost · GM=Σgm or Sales−COGS · GM%=GM/Sales ·
Price=Revenue/Volume · EBITDA=GM−OPEX · Variance=Actual−Base · Variance%=(A−B)/B

## Bridges (reconcile exactly; explicit Other/Residual)
Sales: Volume=(Vol_a−Vol_b)·Price_b; Price=(Price_a−Price_b)·Vol_a; Mix carved from vol.
GM%: pp decomposition — Cost/raw=−Δ(COGS/Sales); Price/vol&mix=total−cost.
EBITDA: Sales impact=(S_a−S_b)·GM%_b; GM impact=(GM%_a−GM%_b)·S_a; OPEX=−(OPEX_a−OPEX_b).

## Advanced attribution (drill-through)
At (customer,product) grain: Volume, Price, Mix, New (only in Actual), Lost (only in
Base), Other. Effect totals equal the sum of their line-level parts (tested).

## Scenario engine (analytics/scenarios.py)
Member levers applied at ROW level: volume_pct scales volume; price_pct scales price;
cost_pct scales unit cost (cost/volume); churn removes the member; scope='all' applies
company-wide; opex_pct / opex_abs flex OPEX (% or absolute £, spread by revenue share).
Revenue/cost/GM are rebuilt from levered volume×price×unit_cost, then Sales/GM/GM%/EBITDA
recomputed by the SAME calc engine → deltas vs baseline are exact and traceable.

### Scenario bridges (baseline actuals → scenario)
scenario_sales_bridge / scenario_gm_pct_bridge / scenario_ebitda_bridge decompose the
baseline→scenario movement and each RECONCILE exactly (tested).

## AI Scenario Agent (ai/scenario_agent.py)
Natural language → structured levers. Two parsers:
- LLM parser: parameter extraction only (returns strict JSON levers). Never calculates.
- Rule-based parser (offline): splits clauses on punctuation/conjunctions, resolves
  member names (fuzzy), detects volume/price/cost %, churn, company-wide scope
  (we/our/company/cost words), and OPEX as % or absolute £ (£500k, £1m).
Clarification: a price/volume change with no member and no company indicator asks the
user for scope before running (scope materially changes the result).

## Budget / Forecast / Projection
Budget: last12m×(1+growth), GM£, COGS with inflation, quarters seasonalised (sum to total).
Forecast: blended growth = run-rate + PY trend + market momentum + budget anchor, seasonal.
Projection: Revenue=base×(1+CAGR)ⁿ; GM%=base±drift·n; EBITDA=GM£−revenue×opex_ratio.
