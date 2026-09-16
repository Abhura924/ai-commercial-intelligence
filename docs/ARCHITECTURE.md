# Architecture

## Scenario flow (redesign)
Natural language → ai/scenario_agent (Scenario Intent Parser: LLM or rule-based) →
Structured Scenario Object (levers + opex + market) → analytics/scenarios
(Deterministic Scenario Engine, reuses calculation_engine) → Scenario Results + Bridges
→ ai/scenario_agent (AI Explanation, deltas only). No second calculation engine.

## Layers (modular; UI decoupled from logic)
Ingestion (ui/pipeline) → RAW storage (data_engine/storage, immutable + audit) →
Semantic mapping → Canonical model → Data quality → Deterministic calc engine →
Bridges/Attribution/Forecast/Budget/Projection/Scenarios (analytics/*) →
AI (ai/analyst, ai/scenario_agent via ai/llm_provider, vendor-neutral) → UI.

## Canonical model (star schema)
FACT_SALES: date, customer, product, region, business_unit, volume, revenue, price,
cost, gross_margin, opex, budget_*, py_*, market_growth. Uploads mapped TO this.

## Storage (cloud-ready)
data/raw (immutable), data/processed (canonical), data/metadata (json + audit_log.jsonl).
storage.py is a thin interface — swap for Postgres/Azure SQL/Fabric/Data Lake later.

## Tech
Python+pandas/numpy (transparent, testable engine) · Streamlit (fast board-grade UI,
replaceable by React/FastAPI without touching analytics/) · Plotly · Parquet · LLM
abstraction (Anthropic/Azure OpenAI/OpenAI).

## Structure
analytics/ (calculation_engine, bridges, attribution, budget, forecast, projection,
scenarios) · data_engine/ · ai/ (analyst, scenario_agent, llm_provider) · ui/ · demo/ ·
tests/ · docs/ · data/
