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

**Methodological limitation (deliberately documented):** the evaluation uses simple,
deterministic checks (keyword containment, numeric tolerance), *not* an LLM-as-judge — this
is 100% reproducible, but does not catch every possible phrasing of a correct answer, and may
miss subtle factual errors. In a production version it would be worth adding a second,
LLM-based "judge" step for the keyword-based questions.

## 6. Load test

```bash
python -m eval.load_test            # 150 synthetic queries, sequential
python -m eval.load_test --num 60 --workers 4   # faster/concurrent variant
```

The script simulates the 4 question types in realistic proportions (≈50% policy, ≈20%
leave_balance, ≈20% ticket, ≈10% out-of-scope/compound) using repeated, generated questions,
and writes the mean/p50/p90/p99 latency — both overall and **broken down per node** — plus
the identified bottleneck into `eval/results/load_test_report.{md,json}`.

**Expected result / bottleneck analysis** (see also the report's generated
`OPTIMIZATION_NOTES` section): the nodes expected to dominate are the LLM-calling ones —
`classify_intent`, `rag.grade_relevance`, `rag.rewrite_query`, `rag.generate_answer`,
`synthesize_response` — since each one is a synchronous Ollama round-trip, and a single
policy question can trigger 3-5 of these calls in sequence. `retrieve` (FAISS) and the
deterministic tools (`calculate_leave`, `create_ticket`) are negligible in cost by comparison
(a couple of milliseconds).

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

## 7. What I actually tested, and this sandbox's limitation

This development environment has **no installed Ollama / downloaded local LLM**, so I was
not able to run the full, real-LLM-driven `run_eval.py`/`load_test.py` live, and I'm not
stating any concrete numbers for their results.

What I actually ran and verified:

- **`pytest tests/` – 16/16 tests green.** `test_tools.py` tests the calculator math and
  SQLite ticket read/write with real code; `test_rag_subgraph.py` tests the retrieve step
  with a **real embedding model and a real FAISS index** (not mocked), while the
  grading/rewrite/generate LLM steps use a deterministic, scripted stub LLM (this is purely
  test infrastructure, not a user-facing mode); `test_graph_routing.py` exercises all 6
  routing branches (policy/calc/ticket/compound/out-of-scope/auto-escalation) through the
  full main graph, and checks the correctness of `trace`/`pending_tasks`/`results`.
- I also ran a real FAISS index build and search over the actual 8 policy documents (see
  `scripts/build_index.py`) with real `sentence-transformers` embeddings – this works, the
  relevant document comes back as the top hit.
- I started the Streamlit UI (`streamlit run`), and it returns HTTP 200 with no error – but I
  could not try the actual chat interaction (which requires Ollama) in a browser.
- I **could not build** the Dockerfile in this sandbox (the Docker daemon isn't running here)
  – I reviewed it manually for correctness, but the actual `docker compose up --build` run
  needs to be done by the user.

**Suggestion:** after `docker compose up --build -d && docker exec policy-assistant-ollama ollama pull llama3.2:3b`,
run `python -m eval.run_eval` and `python -m eval.load_test`, and check the
`eval/results/*.md` reports for the real numbers.

## 8. Known limitations / future improvements

- The evaluation is keyword-based, not LLM-as-judge (see section 5).
- `leave_balance_calculator` computes a single within-year balance; it does not fully
  implement the year-boundary logic of the carryover cap (max 5 days).
- `langchain_community.vectorstores.FAISS` emits a deprecation warning (the package is being
  "sunset") – it still works for now, but a future version should move to a dedicated
  `langchain-faiss` package or another vectorstore.
- The load test runs against a single local Ollama process; measuring real horizontal
  scaling would require multiple model replicas.
