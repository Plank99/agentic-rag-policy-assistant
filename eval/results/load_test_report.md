# Load Test Report

- Queries: 150
- Workers: 1
- Wall clock: 362.5s

## Overall latency (ms)

| mean | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 2416.9 | 2222.2 | 3581.6 | 4180.4 | 4444.1 |

## Per-node latency (ms)

| node | n | mean | p50 | p90 | share of total |
|---|---|---|---|---|---|
| calculate_leave | 38 | 0.0 | 0.0 | 0.0 | 0.0% |
| classify_intent | 150 | 704.6 | 568.2 | 981.9 | 22.9% |
| create_ticket | 30 | 11.6 | 11.1 | 13.0 | 0.1% |
| direct_response | 7 | 916.7 | 927.1 | 945.5 | 1.4% |
| rag.generate_answer | 83 | 857.7 | 688.7 | 1556.8 | 15.4% |
| rag.grade_relevance | 83 | 336.7 | 334.8 | 365.8 | 6.0% |
| rag.retrieve | 83 | 8.5 | 8.1 | 10.1 | 0.2% |
| rag_answer | 83 | 1205.0 | 1028.6 | 1895.1 | 21.6% |
| synthesize_response | 143 | 1047.1 | 1022.6 | 1614.2 | 32.4% |

**Identified bottleneck node:** `synthesize_response`


## Bottleneck analysis

The dominant cost is almost always the LLM-calling nodes (`classify_intent`,
`rag.grade_relevance`, `rag.rewrite_query`, `rag.generate_answer`, `synthesize_response`):
every one of them makes a synchronous round-trip to the local Ollama model, and a single
policy question can trigger 3-5 of these round-trips in sequence. `retrieve` (FAISS) and the
deterministic tools (`calculate_leave`, `create_ticket`) are essentially free by comparison
(single-digit milliseconds), since they never leave the process.

## Optimization suggestions

1. **Collapse LLM round-trips.** `grade_relevance` + `rewrite_query` can be merged into a
   single structured-output call (`{"relevant": bool, "rewritten_query": "..."}`), and
   `grade_relevance` can be skipped entirely when the top FAISS similarity score already
   clears a confidence threshold. That alone removes up to 2 of the 4-5 LLM calls per
   request on the happy path.
2. **Reduce per-call model latency.** Use a smaller/quantized model (or Ollama's
   `num_ctx`/`keep_alive` tuning to avoid repeated context reloads), and batch the
   embedding step for `retrieve` across concurrent requests. For real concurrent load,
   front Ollama with more than one model replica (or a GPU), since a single local Ollama
   process serializes requests regardless of how many client threads call it.
