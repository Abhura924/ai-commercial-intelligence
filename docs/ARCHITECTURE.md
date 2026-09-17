# Architecture

## Unified agent (v0.5)
Natural language → ai/agent_router.classify_intent (LLM or rule-based) → dispatch:
  • scenario  → ai/scenario_agent (levers + horizon) → analytics/forecast.forecast_frame
                (if horizon) → analytics/scenarios (deterministic) → reconciling bridges
  • dashboard → ai/dashboard_builder.build_spec → ui renders blocks from the engine
  • question  → ai/analyst.answer_question (grounded)
The LLM only classifies/extracts/selects — never calculates. No second calc engine.

## Layers
Ingestion (ui/pipeline) → RAW storage (data_engine/storage, immutable + audit) →
Semantic mapping → Canonical model → Data quality → Deterministic calc engine →
Bridges/Attribution/Forecast/Budget/Projection/Scenarios (analytics/*) →
AI (analyst, scenario_agent, dashboard_builder, agent_router via llm_provider) → UI.

## Canonical model (star schema)
FACT_SALES: date, customer, product, region, business_unit, volume, revenue, price,
cost, gross_margin, opex, budget_*, py_*, market_growth. Uploads mapped TO this.

## Storage (cloud-ready)
data/raw (immutable), data/processed (canonical), data/metadata (json + audit_log.jsonl).
storage.py is a thin interface — swap for Postgres/Azure SQL/Fabric/Data Lake later.

## Structure
analytics/ · data_engine/ · ai/ (analyst, scenario_agent, dashboard_builder,
agent_router, llm_provider) · ui/ · demo/ · tests/ · docs/ · data/
