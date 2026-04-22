# MARG — Multi-Agent Research & Report Generator

> A stateful, graph-based multi-agent system that decomposes research into five specialised agent roles, executes iterative self-critique, and delivers fully sourced, structured reports. Built on LangGraph, Ollama, and DuckDuckGo. Runs locally for $0.

---

## The problem with single-pass LLM research

Ask an LLM to research a topic. You get one context window of output, no source verification, no coverage guarantee, and no quality gate. The failure modes are structural — not a model capability problem:

| Failure | Cause | This system's remedy |
|---|---|---|
| Shallow coverage | Single retrieval pass; context window is the ceiling | Planner decomposes topic into N scoped sub-queries; Researcher executes each independently |
| Hallucinated claims | No grounding loop; LLM fills gaps with plausible fabrication | Critic Agent scores factual accuracy; low score triggers targeted re-research, not re-generation from stale context |
| No source attribution | URLs not tracked through the pipeline | Researcher writes `(url, snippet)` tuples to state; Formatter enforces citations per claim |
| No quality gate | Output delivered regardless of quality | Critic scores across 4 dimensions; below-threshold sections route back through the graph |
| Opaque reasoning | Chain-of-thought discarded | Every agent appends its rationale to `state.trace`; full trace returned with report |
| Scope drift | LLM interprets breadth ad hoc | Planner generates a bounded outline; Synthesizer is constrained to it; Critic penalises violations |

---

## How it works

Five agents. One shared state object. One LangGraph state machine.

```
START
  │
  ▼
┌─────────────┐     generates scoped outline + sub-queries per section
│   Planner   │
└──────┬──────┘
       │
  ▼
┌─────────────┐     executes DuckDuckGo searches; scores + maps sources to sections
│  Researcher │ ◄─────────────────────────────────────────────┐
└──────┬──────┘                                               │ FAIL-RETRIEVAL
       │                                                      │ (re-search targeted sections)
  ▼                                                           │
┌─────────────┐     drafts sections from outline + source map │
│ Synthesizer │ ◄──────────────────────────────────┐          │
└──────┬──────┘                                    │          │
       │                                  FAIL-QUALITY        │
  ▼                                       (rewrite with       │
┌─────────────┐     scores 4 dimensions:  critic feedback)    │
│    Critic   │ ────────────────────────────────────┘         │
│  [GATE]     │ ──────────────────────────────────────────────┘
└──────┬──────┘
       │ PASS (or iteration_count ≥ 3)
  ▼
┌─────────────┐     assembles TOC · executive summary · reference list · disclosure
│  Formatter  │
└──────┬──────┘
       │
      END → { report_md, sources[], quality_scores, iteration_count, trace }
```

The critique loop is the architectural centrepiece. The Critic doesn't just return pass/fail — it returns a typed failure (`RETRIEVAL_GAP` | `QUALITY_FAIL`) that routes to the correct agent. A coverage gap routes to the Researcher. A prose quality failure routes to the Synthesizer. Re-searching when the problem is prose quality (and vice versa) wastes iterations and achieves nothing.

---

## Agents

### Planner
Receives the raw query and optional depth/focus parameters. Produces a scoped outline with N sections, relative weights, and 2–4 atomic sub-queries per section. The outline is a constraint, not a suggestion — the Synthesizer cannot introduce sections the Planner did not define.

### Researcher
Executes DuckDuckGo searches per sub-query. Scores each result by cosine similarity against the sub-query using `all-MiniLM-L6-v2` (local, no API cost). Writes structured `(section, url, snippet, relevance)` tuples to state. May be re-invoked for specific sections only — not always a full re-search.

### Synthesizer
Drafts each section from the outline and the corresponding source map entries. Injects inline citation markers at claim level. Constrained to outline structure and source map — cannot invent sources. On re-invocation, receives the Critic's targeted feedback as additional context.

### Critic
The quality gate. Scores the draft across four dimensions with independent thresholds:

| Dimension | Pass threshold | Failure route |
|---|---|---|
| Factual accuracy | ≥ 0.80 | `FAIL-RETRIEVAL` → Researcher |
| Coverage completeness | ≥ 0.75 | `FAIL-RETRIEVAL` → Researcher (targeted sections) |
| Analytical clarity | ≥ 0.70 | `FAIL-QUALITY` → Synthesizer |
| Objectivity | ≥ 0.75 | `FAIL-QUALITY` → Synthesizer |

If `iteration_count` reaches 3, the graph routes to the Formatter unconditionally. Best available output is delivered with a `quality_warning` flag — no blank errors, no infinite loops.

### Formatter
Pure structural transformation. Assembles the final report: executive summary, table of contents, section bodies, numbered reference list, disclosure header. Makes no content decisions.

---

## State schema

All inter-agent communication is through a single `AgentState` TypedDict. No agent passes data through any other channel. The entire execution is serialisable and resumable from any checkpoint.

```python
class AgentState(TypedDict):
    # inputs
    query:           str
    depth:           str                    # "overview" | "standard" | "deep"
    focus_areas:     List[str]

    # planner output
    outline:         List[OutlineSection]   # {title, weight, sub_queries[]}

    # researcher output (append-only — re-search accumulates, not replaces)
    source_map:      Annotated[List[SourceEntry], add_messages]

    # synthesizer output
    draft_sections:  List[dict]             # {title, body, citations[]}
    citation_index:  dict                   # {ref_num → SourceEntry}

    # critic output
    quality_scores:  Optional[QualityScores]
    critic_feedback: Optional[str]          # typed failure + targeted instructions
    iteration_count: int                    # max 3

    # formatter output
    final_report:    Optional[str]          # Markdown

    # observability
    trace:           Annotated[List[str], add_messages]
```

---

## Quickstart

### Local (Docker Compose + Ollama)

```bash
# 1. clone
git clone https://github.com/sidrao2006/marg.git
cd marg

# 2. pull the default model
ollama pull llama3.1:8b

# 3. start the stack
docker compose up -d

# 4. run a research job
curl -X POST http://localhost:8000/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Competitive landscape of vector database providers in 2025",
    "depth": "standard",
    "focus_areas": ["pricing", "performance benchmarks", "managed vs self-hosted"]
  }'
```

Open the Streamlit UI at `http://localhost:8501` for real-time progress and report rendering.

### Cloud Run (hosted)

```bash
gcloud run deploy marg \
  --source . \
  --region europe-west1 \
  --memory 8Gi \
  --cpu 4 \
  --timeout 600 \
  --set-env-vars OLLAMA_MODEL=llama3.1:8b
```

---

## API

### `POST /v1/research`

```json
{
  "query":        "string  — research topic",
  "depth":        "overview | standard | deep",
  "focus_areas":  ["optional", "sub-topics"]
}
```

Returns immediately with a `session_id`. Stream progress via `GET /v1/status/:id` (SSE).

Final response:

```json
{
  "session_id":      "uuid",
  "report_md":       "# Report title\n...",
  "sources":         [{ "ref": 1, "url": "...", "snippet": "..." }],
  "quality_scores":  { "accuracy": 0.85, "coverage": 0.82, "clarity": 0.81, "objectivity": 0.79 },
  "iteration_count": 1,
  "trace":           ["planner:scope defined:0.92", "researcher:28 sources:0.88", "..."]
}
```

---

## Configuration

```yaml
# config.yaml
llm:
  provider:   ollama          # ollama | openai | vertexai
  model:      llama3.1:8b     # any Ollama-compatible model
  base_url:   http://localhost:11434

search:
  provider:   duckduckgo      # duckduckgo | serpapi
  max_results_per_query: 10
  top_k_per_section:     5

critique:
  thresholds:
    accuracy:    0.80
    coverage:    0.75
    clarity:     0.70
    objectivity: 0.75
  max_iterations: 3

output:
  formats: [markdown, pdf]
  include_trace: true
  include_scores: true
```

Switching from Ollama to any OpenAI-compatible endpoint is one config line — no agent code changes required.

---

## Tech stack

| Layer | Component | Notes |
|---|---|---|
| Orchestration | LangGraph 0.2 | `StateGraph` · `TypedDict` state · `SqliteSaver` checkpointer |
| LLM inference | Ollama + Llama 3.1 8B/70B | Local · zero API cost · OpenAI-compatible interface |
| Search | DuckDuckGo wrapper | No API key · no query logging · LangChain tool interface |
| Relevance scoring | `all-MiniLM-L6-v2` | Local sentence-transformers · cosine similarity |
| API | FastAPI + Uvicorn | Async · SSE streaming · Python 3.11 |
| UI | Streamlit | Real-time progress · report rendering |
| Storage | SQLite + local FS | Checkpoints · query log · resumable sessions |
| Deployment | Docker Compose / Cloud Run | Single container · `min-instances=0` on demo tier |

---

## Deployment tiers

| | Demo (local) | Hosted (Cloud Run) | Production (GKE) |
|---|---|---|---|
| LLM | Ollama 8B | Ollama 8B sidecar | Vertex AI Gemini |
| Search | DuckDuckGo | DuckDuckGo | SerpAPI or custom |
| Storage | Local FS | Cloud Storage | Cloud Storage + Firestore |
| Auth | None | Cloud Armor rate-limit | OIDC + Cloud Armor |
| Cost | $0 | GCP free tier | Variable |
| Cold start | Instant | ~10s | < 2s (min-instances=1) |

---

## Resilience

| Scenario | Behaviour |
|---|---|
| Ollama unresponsive | Retry 3× with exponential backoff. All fail → `503` with structured error. No partial report delivered. |
| Search returns no results | Sub-query rewrite attempted once. Still empty → section flagged `source_gap=true`; Critic will route to re-search. |
| Critique loop hits max iterations | Hard route to Formatter. Report delivered with `quality_warning: true` and scores. No blank errors. |
| LLM structured output parse failure | Retry at temperature=0 with explicit JSON schema. Three retries → `ParseError`, state checkpointed for inspection. |

---

## Roadmap

- **PI-1** ✅ Core graph workflow · five-agent baseline · LangGraph state machine
- **PI-2** ✅ Critique loop · typed failure routing · max-iteration guard
- **PI-3** ✅ Structured report formatting · source attribution · disclosure header
- **PI-4** ✅ FastAPI endpoint · Streamlit UI · Docker Compose deployment
- **PI-5** 🔲 Multi-modal input — PDF and image ingestion via document agent
- **PI-6** 🔲 Custom knowledge base — Chroma/FAISS corpus alongside web search
- **PI-7** 🔲 Quantitative Critic scoring — automated hallucination rate metric
- **PI-8** 🔲 Team collaboration — shared session state · concurrent report editing

---

## Project structure

```
marg/
├── agents/
│   ├── base.py          # LLMClient abstraction — swap Ollama/OpenAI/Vertex in one line
│   ├── planner.py
│   ├── researcher.py
│   ├── synthesizer.py
│   ├── critic.py
│   └── formatter.py
├── tools/
│   └── search.py        # DuckDuckGo wrapper + cosine relevance scorer
├── state.py             # AgentState TypedDict — the single source of truth
├── graph.py             # LangGraph StateGraph definition + conditional edges
├── main.py              # FastAPI app · POST /v1/research · GET /v1/status/:id
├── ui/
│   └── app.py           # Streamlit interface
├── config.yaml
├── docker-compose.yml
└── Dockerfile
```

---

## Architecture document

A complete architecture design document — C4 diagrams, agent state contracts, LangGraph graph topology, ADRs, deployment topology, and a live interactive simulator — is available at [`docs/architecture.html`](docs/architecture.html).

---

## License

MIT. See `LICENSE`.

---

*Built by [Siddharth Rao](mailto:rao.siddharth@gmail.com) · TOGAF EA · GCP CA · MLE · Gen AI Leader*
