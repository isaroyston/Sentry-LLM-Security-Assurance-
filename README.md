# Sentry Project

A Red Team & Blue Team project tested on a mock agentic withdrawal chatbot. The objective is to explore the vulnerabilities of agentic architectures through an automated red-teaming app, and to implement multilayer ML + LLM defences against prompt injection, policy circumvention, and scope drift.

The repository ships two cooperating apps: an attacker (the **Red Teaming App**) and a defender (the **Withdrawal Chatbot** wrapped in **Sentry Blue Team** layers). Running them against each other produces a measurable penetration rate that you can iterate on as you tighten guardrails.

---

## Applications

### Red Teaming App

A Streamlit dashboard that runs single-turn and multi-turn adversarial scenarios against any chatbot endpoint, mutates prompts through a library of obfuscation tools (homoglyph, char-swap, payload-mask, etc.), and scores each turn with an LLM judge. Results are written to `attacks/reports/` for diffing across runs.

![Sentry Red Teamer](pictures/sentry_red_teamer.png)

### Withdrawal Chatbot with Sentry Blue Team Layers

A Flask + LangGraph chatbot grounded on official SGBank withdrawal-policy documents. The Blue Team wraps it in three defensive layers: an external **Sentinel** input guardrail, a tool-driven QA agent constrained to approved policy excerpts, and an LLM **output checker** that can rewrite, block, or send the answer back for one retry.

![Withdrawal Chatbot](pictures/withdrawal_chatbot.png)

---

## Project Directory

```
.
├── app.py                          # Flask entry point — withdrawal chatbot UI
├── main.py                         # CLI entry point for the chatbot
├── ingest.py                       # Loads SGBank policy PDFs into the vector store
├── requirements.txt
│
├── src/
│   ├── chatbot/
│   │   ├── withdrawal_chatbot.py   # LangGraph agent, output checker, retry edge
│   │   └── sentinel_guard.py       # External Sentinel input-guardrail client
│   ├── db/
│   │   └── supabase_client.py      # Supabase + pgvector client
│   └── documents/                  # SGBank policy PDFs (RAG source of truth)
│
├── frontend/
│   ├── templates/                  # Flask Jinja templates (index, login, register)
│   └── static/                     # JS + CSS for the chat UI
│
├── attacks/
│   ├── streamlit_app.py            # Red Team dashboard
│   ├── red_teaming.py              # Single-turn attack runner
│   ├── generative_red_team.py      # Multi-turn attacker LLM + scenario walker
│   ├── evaluator.py                # LLM judge for jailbreak success
│   ├── prompt_attacks.py           # Static attack catalogue
│   ├── modules/                    # Mutation tools (homoglyph, char_swap, …)
│   ├── datasets/                   # Bias / toxicity CSVs + multi-turn scenarios
│   └── reports/                    # Penetration test results
│
└── pictures/                       # Architecture diagrams used in this README
```

---

## Run Project Instructions

### Common setup (do this once)

```bash
# 1. Clone + virtualenv
git clone <repo-url> Joint-Intern-Project
cd Joint-Intern-Project
python -m venv venv
source venv/bin/activate                 # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env — required keys:
#   OPENAI_API_KEY=...
#   SENTINEL_API_KEY=...
#   SENTINEL_API_URL=https://sentinel.stg.aiguardian.gov.sg/api/v1/validate
#   SUPABASE_URL=...
#   SUPABASE_KEY=...
```

### Withdrawal Bot

```bash
# 1. One-off: ingest the SGBank policy PDFs into Supabase pgvector
python ingest.py

# 2. Launch the Flask UI
python app.py
# → http://localhost:5000
```

For a headless CLI session: `python main.py`.

### Red Team App

The Red Team app talks to the chatbot over HTTP, so you need the bot's API endpoint running before you launch the dashboard.

```bash
# 1. Start the FastAPI endpoint that exposes the chatbot to the attacker
uvicorn api:app --host 0.0.0.0 --port 8000
# → http://localhost:8000  (POST /chat)

# 2. In a second terminal, launch the Streamlit dashboard
streamlit run attacks/streamlit_app.py
```

The dashboard loads `attacks/.env` if present, otherwise it falls back to the repo-root `.env`. Point its `CHATBOT_API_URL` (or equivalent setting in the dashboard) at the `api.py` endpoint above. Reports are written to `attacks/reports/` and can be replayed or diffed between defence iterations.

---

## Red Team

![Red Team Workflow](pictures/red_teamer_workflow.png)

The attacker LLM is briefed with a scenario `objective`, a soft `prompt_reference` for the next scripted turn, and the running conversation history. Each generated prompt may be passed through a chain of mutation tools before it hits the bot, and every bot response is scored by an LLM evaluator that flags jailbreak success when the response leaks internal state, drifts off-topic, or surrenders policy-circumventing detail.

## Blue Team

![Blue Team Workflow](pictures/blue_team_workflow.png)

Four layered defences sit in front of the QA agent:

1. **Language query filter** — a per-character allowlist guard (`block_non_english` in `withdrawal_chatbot.py`) that rejects any input containing CJK characters, Cyrillic, Arabic, or other non-Latin scripts before it reaches the LLM. This blocks homoglyph and foreign-script prompt-injection attacks (e.g. Chinese-character payloads, Cyrillic look-alikes used to spell ASCII tokens).
2. **Sentinel input guardrail** — external safety classifier that blocks prompt-injection and unsafe input before any tool is reached.
3. **Tool-driven QA agent** — answers are grounded on `get_account_balance`, `get_withdrawal_limit`, and a RAG `policy_checker` that retrieves only from approved SGBank policy documents.
4. **LLM output checker** — validates every draft for safety, scope, and compliance. It can return the draft as-is, rewrite it, or route it back to the QA agent for one retry with feedback.

## Withdrawal Chatbot

![Withdrawal Chatbot Workflow](pictures/withdrawal_bot_workflow.png)

The chatbot is implemented as a LangGraph state machine: `load_history → sentinel_input → qa_agent → output_check`, with a single conditional edge from `output_check` back to `qa_agent` when a retry is requested. Conversation history and per-user account snapshots live in Supabase; policy excerpts live in pgvector and are retrieved through a keyword-rewrite + cosine-similarity cache pipeline.

---

## Environment Variables

```env
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=384

SENTINEL_API_KEY=
SENTINEL_API_URL=https://sentinel.stg.aiguardian.gov.sg/api/v1/validate

SUPABASE_URL=
SUPABASE_KEY=

RED_TEAM_EVAL_MODEL=gpt-4o-mini
```
