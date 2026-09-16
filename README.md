# AI Commercial Intelligence — v0.3.2

An AI-powered **FP&A & Commercial Analyst** prototype. Upload business data (no rigid
template); the platform understands it, standardises it, calculates the commercial
story with a **deterministic engine**, and explains it.

> The LLM never performs financial arithmetic. It only understands intent, extracts
> scenario assumptions, and explains results. Every calculation, bridge and
> reconciliation is done by the deterministic engine.

## Run it (from THIS folder — the project root)
```
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run ui/app.py
python -m pytest tests -q        # 21 passed — bridges, attribution & scenarios reconcile
```
Opens at http://localhost:8501 in Demo mode. Enable LLM understanding with
ANTHROPIC_API_KEY / AZURE_OPENAI_* / OPENAI_API_KEY (a rule-based parser works offline).

## What's new — AI Scenario Analyst (redesign)
The old slider-based "Scenario Lab" is replaced by a **natural-language AI Scenario Analyst**:
1. Type a scenario in plain English — *"What happens if Customer A grows volume 15%,
   prices +3%, raw material cost +5% and OPEX +£500k?"*
2. **Scenario understood** — the AI extracts structured assumptions (shown in a table);
   you can **edit & re-run**.
3. **Scenario impact** — Sales / GM / GM% / EBITDA current→scenario KPI cards.
4. **Bridges** — Sales, GM% and EBITDA bridges that **reconcile exactly**.
5. **AI management commentary** — WHAT / WHY / SO-WHAT, grounded only in the deltas.
6. **Follow-up chat** — modifies the scenario and re-runs the engine.
7. **Scenario history + comparison** — save, compare side by side, delete.
8. **Clarification** — a bare *"increase price 5%"* asks whether it's company-wide or a
   specific customer/product (scope materially changes the answer).

Architecture: `Natural Language → ai/scenario_agent (intent parser, LLM or rule-based)
→ structured levers → analytics/scenarios (deterministic engine) → results + bridges
→ AI explanation`. The existing calculation engine is reused — there is **no second
calculation engine**.

## Pages
Executive Dashboard · Sales & Drivers · **AI Scenario Analyst** · Forecast · Budget ·
5-Year Projection · Bridges · AI Analyst · Data & Quality

See docs/ARCHITECTURE.md · docs/METHODOLOGY.md · docs/ROADMAP.md.
