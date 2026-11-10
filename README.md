# Agentic RAG Chatbot – Internal Company Policy Assistant

Agentic RAG chatbot prototype in Python, built on
[LangGraph](https://langchain-ai.github.io/langgraph/). Uses exclusively local, open-source
models (Ollama LLM + sentence-transformers embeddings) — no paid API required.

## 1. Problem statement and goal

**Domain:** an internal HR/IT/Finance/Legal policy assistant for a large company. Employees
frequently ask repetitive, basic questions ("how many vacation days do I have?", "what's the
gift limit?", "how do I request VPN access?"), which currently consume HR/IT ticket capacity
and colleagues' time.

- **Why is it relevant?** These questions occur at high volume but low complexity — a
  textbook retrieval-augmented automation case.
- **What user need does it meet?** The employee gets an instant, source-backed answer without
  having to read the entire policy document or open a ticket for a simple question.
- **Why agentic RAG?** The questions vary in type (factual lookup in policy docs, numeric
  calculation, concrete action/ticket creation), and often appear *combined* in a single
  message. A plain "retrieve-then-generate" RAG pipeline cannot handle this: an agent is
  needed that decides *what* needs to be done, *decomposes it into subtasks*, executes them
  independently, and — if the RAG lookup doesn't produce a confident answer — **autonomously
  escalates** by opening a support ticket.

## 2. Architecture

### 2.1 Main LangGraph graph (`app/graph.py`) – 6 nodes

```
                     ┌─────────────────┐
                     │ classify_intent │  (LLM: task decomposition,
                     └────────┬────────┘   JSON: tasks[], rag_query, calc_args, ticket_args)
                              │
              ┌───────────────┼────────────────────────┐
              │ (tasks==[])   │ (tasks[0]==...)         │
              ▼               ▼                         ▼
      ┌───────────────┐ ┌───────────┐ ┌───────────────┐ ┌───────────────┐
      │direct_response│ │rag_answer │ │calculate_leave│ │ create_ticket │
      └───────┬───────┘ └─────┬─────┘ └──────┬────────┘ └──────┬────────┘
              │               │ (RAG subgraph, see below)       │
              │               │ low confidence → auto-escalate  │
              │               └────────────┬─────────┬──────────┘
              │                            (dispatcher: any pending_task left?)
              │                                       │
              ▼                                       ▼
             END                          ┌─────────────────────┐
                                           │ synthesize_response │
                                           └──────────┬───────────┘
                                                      ▼
                                                     END
```

| Node | Role |
|---|---|
| `classify_intent` | Entry point. Asks the LLM for structured JSON: which subtasks are needed (`policy_question`, `leave_balance`, `create_ticket`), and extracts the required parameters. **This is the task decomposition step.** |
| `direct_response` | If there is no relevant task (greeting / out-of-scope question) — a short reply with no retrieval. |
| `rag_answer` | Calls the dedicated RAG subgraph (see 2.2). If the subgraph signals low confidence, it **autonomously** adds a `create_ticket` task to the queue and generates escalation ticket parameters — a concrete instance of **autonomous decision-making**. |
| `calculate_leave` | Calls the deterministic PTO-balance calculator tool. |
| `create_ticket` | Calls the support-ticket tool (user-requested or auto-escalated). |
| `synthesize_response` | Combines the sub-results accumulated in the `results` state into one coherent final answer, with source citations / ticket IDs. |

**Routing:** after every task node, a router function decides whether there is still a
`pending_task` in the queue (if so, go to the next task node), or all tasks have run (then go
to `synthesize_response`). This is the **conditional routing**. `AgentState`
(`app/state.py`) carries the intermediate results (`results` dict), the task queue
(`pending_tasks`), the escalation flag, and a full `trace` of every node execution (node
name, short decision description, runtime) — this is the **state management**.

### 2.2 RAG subgraph (`app/rag/subgraph.py`) – separate, modular sub-graph

Not counted among the main graph's 6 nodes; it is a standalone, compiled `StateGraph` that
the `rag_answer` node calls with a query, returning an `{answer, sources, confidence, trace}`
structure. It follows the **corrective-RAG** pattern with a self-correction loop:

```
retrieve → grade_relevance ──relevant──→ generate_answer → (end)
              │
              └──not relevant & retries left──→ rewrite_query → retrieve (loop back)
              └──not relevant & retries exhausted──→ generate_answer (confidence="low")
```

1. **`retrieve`** – FAISS similarity search (`k=4`) using the local `all-MiniLM-L6-v2`
   embedding model.
2. **`grade_relevance`** – the LLM decides whether the retrieved snippets are sufficient.
3. **`rewrite_query`** – if not, the LLM rewrites the question (e.g. resolves pronouns like
   "this"/"that", sharpens keywords), and returns to `retrieve` — at most 2 times.
4. **`generate_answer`** – generates an answer with source citations, or — if the retries are
   exhausted — returns with `confidence="low"`, which triggers auto-escalation in the main
   graph.

Since this is an independently compiled graph, it can be dropped into any other agent
unchanged — this is what makes it a modular, reusable RAG subsystem.

### 2.3 Tools (`app/tools/`)

1. **`leave_balance_calculator`** (`calculator.py`) – deterministic Python logic (1.75
   days/month accrual = 21 days/year, per `pto_policy.md`), wrapped in `@tool`. Not
   retrieval-based; auditable, since the math never comes from the LLM.
2. **`create_support_ticket`** (`ticketing.py`) – a side-effecting tool that writes to
   SQLite, simulating a real ticketing system (ticket ID, category, priority, status). Called
   both by explicit user request *and* by the RAG escalation logic.

### 2.4 Data source

8 synthetic but realistic company policy documents (`app/data/policies/*.md`): IT Security,
Remote Work, Expense Reimbursement, PTO, Data Protection/GDPR, Code of Conduct,
Equipment/BYOD, Travel. The emphasis is on quality of processing:

- `app/rag/ingest.py` – markdown loading, `RecursiveCharacterTextSplitter` (≈500/50 chunk
  size/overlap), source metadata on every chunk.
- `app/rag/vectorstore.py` – rebuilds the index only when the corpus hash changes
  (`corpus.fingerprint`), with FAISS persistence into `app/data/index/`.
- **Scalability:** the storage layer is isolated into a single module (`vectorstore.py`); for
  a larger corpus, switching to a managed vector DB (pgvector, Pinecone, etc.) only touches
  this file — the ingestion/subgraph code stays unchanged.

## 3. Model choice and trade-offs

| Component | Choice | Rationale |
|---|---|---|
| LLM | **Ollama, `llama3.2:3b`** (configurable via `OLLAMA_MODEL` env var) | No paid API. A 3B model runs at acceptable speed even on a laptop CPU, which is enough for the demo. Trade-off: a smaller model is less reliable at following a strict JSON format and weaker at multi-step reasoning → so `app/llm.py` defensively parses the JSON response (extracts the first `{...}` block via regex, falls back to a sensible default on a missing/invalid field), and the prompts give explicit, few-step instructions. A larger model (e.g. `qwen2.5:14b`) would give better quality, but is significantly slower on CPU; worth switching if a GPU is available. |
| Embedding | **`sentence-transformers/all-MiniLM-L6-v2`** (local, runs on CPU) | Small (~90 MB), fast, no API key, good enough retrieval quality for a corpus of this size. |
| Vectorstore | **FAISS** (in-process, persisted to disk) | No separate infrastructure needed for a prototype; see the 2.4 scalability note. |

## 4. Installation and running

### 4.1 Docker Compose (recommended)

```bash
cp .env.example .env   # optional, defaults also work
docker compose up --build -d
docker exec policy-assistant-ollama ollama pull llama3.2:3b   # required on first startup
```

The UI is then available at `http://localhost:8501`. `docker-compose.yml` starts two
services: `ollama` (model server, with a persistent volume) and `app` (the Streamlit UI,
which reaches it via `OLLAMA_BASE_URL=http://ollama:11434`).

> **Note:** even `llama3.2:3b` can be slow on a CPU-only host (see the load test results
> below) – if a GPU-backed Ollama host is available, point `OLLAMA_BASE_URL` at it, or choose
> a smaller model (`llama3.2:1b`).

### 4.2 Running locally (venv)

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows; Linux/Mac: .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b     # requires a separately installed Ollama: https://ollama.com
streamlit run ui/streamlit_app.py
```

The index is built automatically on first startup (cached into `app/data/index/`); to force
a rebuild: `python scripts/build_index.py --force`.

### 4.3 Tests

```bash
pytest tests/ -v
```

## 5. Functional evaluation

`eval/eval_questions.json` – 16 questions covering all 4 task types: 6 pure
`policy_question` entries (one per document, plus one deliberately ambiguous phrasing that
exercises the RAG subgraph's query-rewrite loop), 3 `leave_balance` (with explicit dates so
the ground truth can be computed deterministically), 3 `create_ticket`, 2 `compound`
(multiple subtasks in one question), 1 `out_of_scope`.

```bash
python -m eval.run_eval
```

This runs all 16 questions against the **real, Ollama-backed** graph and automatically
grades them: keyword match (policy answers), numeric tolerance (calculator), category match
(tickets), route match (out-of-scope). Output: `eval/results/eval_report.{md,json}`, with
per-node average/max latency.

**Real result (`llama3.2:3b`, Docker Compose stack, 2026-11-07): 11/16 (68.8%).**

| node | count | avg_ms | max_ms |
|---|---|---|---|
| classify_intent | 16 | 2860.0 | 4375.6 |
| rag.retrieve | 13 | 10.0 | 15.0 |
| rag.grade_relevance | 13 | 3359.9 | 4535.1 |
| rag.rewrite_query | 4 | 2807.2 | 4065.1 |
| rag.generate_answer | 9 | 3911.9 | 6424.5 |
| rag_answer | 9 | 10030.0 | 19861.9 |
| calculate_leave | 4 | 0.0 | 0.0 |
| create_ticket | 5 | 13.2 | 23.2 |
| direct_response | 1 | 3893.8 | 3893.8 |
| synthesize_response | 15 | 6640.2 | 12956.6 |

5 questions failed, for two very different reasons:

- **q5 (real failure):** "Under GDPR, how quickly must the company respond to a data
  subject access request?" – the model didn't reproduce the exact "30 calendar days" fact in
  its answer, even though `data_protection_policy.md` contains it. This is a genuine
  small-model quality limitation (the generated answer came out too generic), not a routing
  or retrieval failure — the correct document was retrieved into the top-k.
- **q10, q11, q12, q14 (grading too strict, not an agent failure):** in all 4 cases the
  ticket was **created correctly**, with the right `summary`/`priority` fields (e.g. q10:
  `{"category": "it", "summary": "lost company laptop", "priority": "high"}`), but the
  `category` field didn't literally match the expected string recorded in
  `eval_questions.json` (`"it_equipment_request"`). `_CLASSIFY_PROMPT` deliberately asks the
  model for a free-form `snake_case` category (no fixed enum), so the model returns its own,
  reasonable category name (`"it"`, `"HR"`, `"expense_dispute"`, `"security_incident"`) —
  which is functionally correct, just not accepted by exact-match grading. **Lesson:** in a
  production version, the category should either be constrained to a closed enum in the
  classify prompt, or the grading should be keyword-/synonym-based instead of exact match.

**Methodological limitation (deliberately documented):** the evaluation uses simple,
deterministic checks (keyword containment, numeric tolerance, exact-match category), *not*
an LLM-as-judge — this is 100% reproducible, but — as the q10-q12/q14 cases above show — it
can sometimes produce a false negative on a functionally correct answer. In a production
version it would be worth adding a second, LLM-based "judge" step for the keyword/category
questions.

## 6. Load test

```bash
python -m eval.load_test            # 150 synthetic queries, sequential
python -m eval.load_test --num 60 --workers 4   # faster/concurrent variant
```

The script simulates the 4 question types in realistic proportions (≈50% policy, ≈20%
leave_balance, ≈20% ticket, ≈10% out-of-scope/compound) using repeated, generated questions,
and writes the mean/p50/p90/p99 latency — both overall and **broken down per node** — plus
the identified bottleneck into `eval/results/load_test_report.{md,json}`.

**Real result (`llama3.2:3b`, Docker Compose stack, 150 queries, 1 worker, total wall clock
1565.6s ≈ 26 min):**

| mean | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 10437.3 ms | 9562.4 ms | 15241.9 ms | 23827.7 ms | 25392.5 ms |

| node | n | mean_ms | p90_ms | share of total |
|---|---|---|---|---|
| synthesize_response | 143 | 5677.7 | 8711.7 | **43.3%** |
| classify_intent | 150 | 2770.9 | 3711.1 | 22.2% |
| rag_answer | 83 | 3736.7 | 6860.0 | 16.5% |
| rag.generate_answer | 83 | 1795.7 | 4798.3 | 7.9% |
| rag.rewrite_query | 30 | 2860.6 | 3358.4 | 4.6% |
| rag.grade_relevance | 113 | 654.4 | 436.2 | 3.9% |
| direct_response | 7 | 3834.6 | 4336.1 | 1.4% |
| create_ticket | 45 | 15.8 | 23.2 | ~0.0% |
| rag.retrieve | 113 | 9.9 | 11.7 | ~0.0% |
| calculate_leave | 38 | 0.0 | 0.0 | 0.0% |

**Identified bottleneck node: `synthesize_response`** (the report script flags this
automatically) – this is a single, relatively long-context LLM call that runs at the end of
almost every request (143/150 runs), so in aggregate it dominates nearly half of the total
time, more than `classify_intent` or the sum of the several smaller calls in the RAG chain.
This confirms the original hypothesis (synchronous LLM round-trips dominate), just not where
we'd expect at first — it's not the longest call chain (the RAG rewrite loop) that is most
expensive, but the single, large-context call that stitches all sub-results together at the
end. `retrieve` (FAISS) and the deterministic tools (`calculate_leave`, `create_ticket`) are,
as expected, negligible in cost (a couple of milliseconds).

**Two concrete optimization suggestions:**
1. **Collapse LLM round-trips.** `grade_relevance` + `rewrite_query` can be merged into a
   single structured-output call (`{"relevant": bool, "rewritten_query": "..."}`), and
   `grade_relevance` can be skipped entirely when the top FAISS similarity score already
   clears a confidence threshold. That alone removes up to 2 of the 4-5 LLM calls per
   request on the happy path.
2. **Reduce per-call latency / add concurrency.** A smaller/quantized model, or tuning
   Ollama's `keep_alive`/`num_ctx` to avoid repeated context reloads; real concurrent load
   needs more than one Ollama replica or a GPU, since a single local Ollama process
   serializes requests regardless of how many client threads call it.

## 7. What I actually tested

This sandbox did not have Docker or a local Ollama model available while the graph/RAG/tool
logic was first being built, so the structure was initially validated with deterministic
stub-LLM tests. Once the user brought up the Docker Compose stack, the system was tested
live against a real Ollama-backed model (`llama3.2:3b`). What actually ran:

- **`pytest tests/` – 16/16 tests green.** `test_tools.py` tests the calculator math and
  SQLite ticket read/write with real code; `test_rag_subgraph.py` tests the retrieve step
  with a **real embedding model and a real FAISS index** (not mocked), while the
  grading/rewrite/generate LLM steps use a deterministic, scripted stub LLM (this is purely
  test infrastructure, not a user-facing mode); `test_graph_routing.py` exercises all 6
  routing branches through the full main graph.
- **`docker compose up --build`** actually ran and brought up both services
  (`policy-assistant-ollama`, `policy-assistant-app`); the Streamlit UI is live at
  `http://localhost:8501`.
- **The full 16-question functional eval (`eval/run_eval.py`) and the 150-query load test
  (`eval/load_test.py`) both actually ran against the real `llama3.2:3b` model** – see
  sections 5 and 6 for the real numbers.
- **Found and fixed two real bugs while testing against the live model:** (1)
  `_CLASSIFY_PROMPT` contained a literal `"YYYY-MM-DD"` placeholder that `llama3.2:3b` echoed
  back verbatim, causing the leave-balance calculator to fail on a plain policy question;
  replaced it with a concrete sample date. (2) `_SYNTHESIZE_PROMPT` didn't explicitly forbid
  inventing a source URL, so the model hallucinated a fake hyperlink for an internal filename
  citation; fixed with an explicit instruction not to invent URLs.

## 8. Known limitations / future improvements

- The evaluation is keyword-based, not LLM-as-judge (see section 5).
- `leave_balance_calculator` computes a single within-year balance; it does not fully
  implement the year-boundary logic of the carryover cap (max 5 days).
- `langchain_community.vectorstores.FAISS` emits a deprecation warning (the package is being
  "sunset") – it still works for now, but a future version should move to a dedicated
  `langchain-faiss` package or another vectorstore.
- The load test runs against a single local Ollama process; measuring real horizontal
  scaling would require multiple model replicas.
