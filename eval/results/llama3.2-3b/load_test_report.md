# Load Test Report

- Queries: 150
- Workers: 1
- Wall clock: 1565.6s

## Overall latency (ms)

| mean | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 10437.3 | 9562.4 | 15241.9 | 23827.7 | 25392.5 |

## Per-node latency (ms)

| node | n | mean | p50 | p90 | share of total |
|---|---|---|---|---|---|
| calculate_leave | 38 | 0.0 | 0.0 | 0.0 | 0.0% |
| classify_intent | 150 | 2770.9 | 3104.2 | 3711.1 | 22.2% |
| create_ticket | 45 | 15.8 | 12.5 | 23.2 | 0.0% |
| direct_response | 7 | 3834.6 | 3886.6 | 4336.1 | 1.4% |
| rag.generate_answer | 83 | 1795.7 | 1181.6 | 4798.3 | 7.9% |
| rag.grade_relevance | 113 | 654.4 | 394.7 | 436.2 | 3.9% |
| rag.retrieve | 113 | 9.9 | 9.0 | 11.7 | 0.1% |
| rag.rewrite_query | 30 | 2860.6 | 3064.8 | 3358.4 | 4.6% |
| rag_answer | 83 | 3736.7 | 1672.4 | 6860.0 | 16.5% |
| synthesize_response | 143 | 5677.7 | 5551.1 | 8711.7 | 43.3% |

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
