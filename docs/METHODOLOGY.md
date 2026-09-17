# Calculation, Bridge, Scenario & Forecast Methodology

All figures come from analytics/* (deterministic). The LLM never does arithmetic.

## Core
Sales=Σrevenue · COGS=Σcost · GM=Sales−COGS · GM%=GM/Sales · Price=Revenue/Volume ·
EBITDA=GM−OPEX · Variance=Actual−Base.

## Bridges (reconcile exactly; explicit Other/Residual)
Sales: Volume=(Vol_a−Vol_b)·Price_b; Price=(Price_a−Price_b)·Vol_a; Mix carved from vol.
GM%: pp decomposition — Cost/raw=−Δ(COGS/Sales); Price/vol&mix=total−cost.
EBITDA: Sales impact=(S_a−S_b)·GM%_b; GM impact=(GM%_a−GM%_b)·S_a; OPEX=−(OPEX_a−OPEX_b).

## Scenario engine
Member levers at ROW level (volume/price/cost %, churn, scope='all', OPEX %/absolute).
Revenue/cost/GM rebuilt from levered volume×price×unit_cost; Sales/GM/GM%/EBITDA
recomputed by the SAME calc engine. Scenario bridges reconcile baseline→scenario.

## v0.4 Forecast horizon (analytics/forecast.forecast_frame)
Rolls the latest 12 months forward `horizon` months at (customer,product) grain:
each line keeps price/unit-cost/GM; volume grows by the blended monthly growth
(run-rate + PY trend + market momentum + budget anchor) with prior-year seasonality.
The un-levered forecast is stored as the period's budget comparator. A scenario run
over this frame therefore recalculates through the same engine and its bridges still
reconcile — the impact is measured against the FORECAST baseline, not trailing actuals.

## v0.4 Board Builder (ai/dashboard_builder)
NL request → dashboard SPEC (ordered list of block ids) chosen by the LLM or a
rule-based keyword matcher. Only block SELECTION is AI; each block renders numbers
from the deterministic engine.
