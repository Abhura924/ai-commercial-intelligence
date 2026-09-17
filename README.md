# AI Commercial Intelligence — v0.5

An AI-powered **FP&A & Commercial Analyst** prototype. Upload business data (no rigid
template); the platform understands it, standardises it, calculates the commercial
story with a **deterministic engine**, and explains it.

> The LLM never performs financial arithmetic. It classifies intent, extracts scenario
> assumptions, and picks dashboard blocks. Every calculation, bridge and reconciliation
> is done by the deterministic engine.

## Run it (from THIS folder — the project root)
```
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run ui/app.py
python -m pytest tests -q        # 26 passed
```
Opens at http://localhost:8501 in Demo mode. Enable LLM understanding with
ANTHROPIC_API_KEY / AZURE_OPENAI_* / OPENAI_API_KEY (a rule-based parser works offline).

## What's new in v0.5 — One unified AI Analyst Agent
The separate "AI Scenario Analyst", "Board Builder" and "AI Analyst" tabs are merged
into a **single AI Analyst Agent**. Describe what you need in one box and the agent
routes automatically:
- **Scenario** — "what if Customer A grows 15%, prices +3%, over the next 6 months" →
  runs the deterministic scenario engine (with the forecast-horizon selector) and shows
  impact, bridges and commentary.
- **Dashboard** — "build me a board pack with 5 KPIs, a sales bridge and the top 3
  issues" → assembles a Board Pack from the right blocks.
- **Question** — "why is sales below budget?" → a grounded analytical answer.

`ai/agent_router.py` classifies the request (LLM or rule-based) and dispatches to the
right capability. It gracefully falls back (e.g. an unparseable "scenario" that looks
like a view becomes a dashboard; otherwise a question). Ambiguous changes (e.g. bare
"increase price 5%") still ask a clarifying question.

## Pages
Executive Dashboard · **AI Analyst Agent** · Sales & Drivers · Forecast · Budget ·
5-Year Projection · Bridges · Data & Quality

See docs/ARCHITECTURE.md · docs/METHODOLOGY.md · docs/ROADMAP.md.
