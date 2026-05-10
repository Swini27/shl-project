# SHL Assessment Recommender — System Documentation

## 1. Overview

This project is a **stateless conversational AI agent** built with FastAPI and the Google Gemini API. It helps recruiters discover relevant SHL assessments through natural conversation. Instead of a traditional keyword search or a simple form-fill UI, the agent understands recruiter *intent* across a multi-turn dialogue and recommends matching assessments from SHL's test catalog.

### Core Design Principles

| Principle | Implementation |
|---|---|
| **Stateless API** | Every `POST /chat` request carries the full conversation history. No per-session data is stored on the server. |
| **Zero Hallucination** | Recommendations are validated against a local copy of the SHL catalog. The LLM cannot invent assessment names. |
| **Separation of Concerns** | LLM is used only for *understanding intent*. A deterministic Python router + a vector DB handles *retrieval*. |
| **Graceful Degradation** | Failures at any stage return a human-readable fallback response rather than a 500 error to the end user. |

---

## 2. Architecture

```
Client (recruiter)
        │
        │ POST /chat
        │ { "messages": [ entire conversation history ] }
        ▼
┌───────────────────────────────────────────────────────────────────┐
│                         FastAPI  (main.py)                        │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Step 1: State Extraction                                   │  │
│  │  Gemini 2.5 Flash reads the full conversation history.      │  │
│  │  Returns a structured JSON blob (ConversationState):        │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │  user_intent: CLARIFYING | READY_TO_RECOMMEND | ...  │   │  │
│  │  │  job_role, seniority, technical_stack, soft_skills   │   │  │
│  │  │  reply_to_user, is_fulfilled                         │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Step 2: Deterministic Python Router                        │  │
│  │  routes based on user_intent:                               │  │
│  │  ┌─────────────────────────────────────────────────────┐    │  │
│  │  │ OFF_TOPIC      → Hard guardrail reply               │    │  │
│  │  │ CLARIFYING     → Return LLM's clarifying question   │    │  │
│  │  │ READY/REFINING → Vector search (ChromaDB) → filter  │    │  │
│  │  │ COMPARE        → RAG + Gemini grounded comparison   │    │  │
│  │  └─────────────────────────────────────────────────────┘    │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
        │
        │ ChatResponse { reply, recommendations[], end_of_conversation }
        ▼
      Client
```

---

## 3. Component Deep Dive

### 3.1 State Extraction (LLM Layer)

**File:** `main.py` | **Class:** `ConversationState`

This is the heart of the agent. The entire conversation history is sent to `gemini-2.5-flash` with `temperature=0.0` (fully deterministic). The model is asked to output a **structured JSON object** (not free text) that is schema-validated by Pydantic.

```python
class ConversationState(BaseModel):
    user_intent: str          # One of 5 states (see below)
    job_role: Optional[str]   # e.g., "Software Engineer"
    seniority: Optional[str]  # e.g., "entry-level", "manager"
    technical_stack: List[str] # e.g., ["Python", "AWS"]
    soft_skills: List[str]     # e.g., ["leadership", "communication"]
    assessment_types: List[str] # e.g., ["personality", "coding"]
    other_constraints: List[str] # e.g., ["English only", "< 30 mins"]
    is_fulfilled: bool         # True when the user's need is met
    reply_to_user: str         # Gemini's text reply to the recruiter
```

**Why structured output?** Asking for a free-text reply and then parsing it is fragile. By enforcing a JSON schema via `response_mime_type="application/json"` and `response_schema=ConversationState`, Gemini is constrained to produce a well-formed, predictable object that Python can validate and act on.

**Cumulative State Reconstruction:** The system instruction tells Gemini to populate constraints from **all** user messages in the history, not just the latest one. This is how the agent "remembers" — not via a server-side database, but by re-reading the entire transcript on every call.

---

### 3.2 Intent Classification & Routing

**File:** `main.py` | **Function:** `chat_endpoint`

The extracted `user_intent` field drives a deterministic Python `if/elif` chain — not another LLM call.

| Intent | Trigger | Action |
|---|---|---|
| `CLARIFYING` | Vague request, not enough info to search | Return Gemini's clarifying question, zero recommendations |
| `READY_TO_RECOMMEND` | Enough context to form a shortlist | Vector search → validate → return top 10 |
| `REFINING` | User adds/changes constraints mid-conversation | Same as READY_TO_RECOMMEND, but Gemini updates constraints cumulatively |
| `COMPARE` | User asks to compare specific tests | RAG lookup → Gemini writes a grounded comparison paragraph |
| `OFF_TOPIC` | Request outside assessment scope | Hard-coded guardrail reply, no LLM call |

**Why deterministic routing?** Using an LLM to decide what to do with the LLM's output adds latency and non-determinism to business logic that can be expressed perfectly in code. The LLM is used where it excels (understanding language); Python is used where it excels (reliable, fast branching).

---

### 3.3 RAG Pipeline (Retrieval Layer)

**File:** `rag_pipeline.py` | **Class:** `RAGCatalog`

This component handles the local SHL test catalog. It uses ChromaDB as a local vector database.

#### Startup (Index Building)

On first run, the `RAGCatalog` class:
1. Loads the full catalog from `catalog.json` into memory as a Python dict (`catalog_map`), keyed by `entity_id` for instant lookups.
2. Builds a ChromaDB vector index. Each document in the index is a rich text string composed of the assessment's name, description, job levels, categories, duration and languages.
3. Persists the index to disk (`/chroma_db/`). Subsequent restarts re-use the existing index without rebuilding.

#### Query Construction

The `search()` method translates the structured `ConversationState` into a plain-English query string that is semantically meaningful for vector similarity search:

```
Role: Software Engineer
Level: Mid-level
Tech Stack: Python, AWS
Soft Skills: leadership
Test Type: coding
```

This is superior to a keyword search because semantically related terms (e.g., "cloud" matching "AWS development") can be matched.

#### Validation (No Hallucination Guarantee)

After retrieval, every result is cross-referenced against the in-memory `catalog_map`. If a `doc_id` returned by ChromaDB is not found in the map, it is silently skipped. This ensures that the API **can only ever return assessments that exist in the ground-truth `catalog.json`**.

---

### 3.4 API Contract

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Returns `{"status": "ok"}`. Used for uptime monitoring. |
| `POST` | `/chat` | Main conversational endpoint. |

**Request Schema (`POST /chat`):**
```json
{
  "messages": [
    { "role": "user", "content": "I need to assess a software engineer." },
    { "role": "assistant", "content": "What seniority level?" },
    { "role": "user", "content": "Mid-level, focused on Python and AWS." }
  ]
}
```

**Response Schema:**
```json
{
  "reply": "Here are the top assessments for a mid-level Python/AWS engineer...",
  "recommendations": [
    {
      "name": "Amazon Web Services (AWS) Development (New)",
      "url": "https://www.shl.com/products/product-catalog/...",
      "test_type": "Knowledge & Skills"
    }
  ],
  "end_of_conversation": false
}
```

> [!IMPORTANT]
> **The API is fully stateless.** Clients must send the **complete conversation history** in every request. The server stores no session state.

---

## 4. Anti-Hallucination Strategy

This system implements a layered approach to prevent the LLM from inventing assessments:

1. **Structured Output (Gemini):** The state extraction call produces a structured `ConversationState`. Gemini is explicitly instructed in the system prompt: *"Never hallucinate assessment names or capabilities."*
2. **LLM Does Not Name Assessments:** Gemini never writes assessment names in the `reply_to_user` field. It only classifies intent and extracts constraints. Assessment names come exclusively from the vector DB query results.
3. **Double Validation:** Every ID returned by ChromaDB is validated against the in-memory `catalog_map` before being surfaced in the API response.
4. **Grounded Comparison:** When comparing assessments (`COMPARE` intent), the relevant catalog entries are injected directly into the Gemini prompt as context. Gemini is explicitly instructed to base its comparison *only* on that data and to state if an assessment is not found.

---

## 5. Setup & Running

### Prerequisites
- Python 3.9+
- A Google Gemini API key (from [aistudio.google.com](https://aistudio.google.com/app/apikey))

### Installation

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
echo "GEMINI_API=your_api_key_here" > .env

# 4. (Optional) Delete old ChromaDB index if catalog.json has changed
rm -rf chroma_db/

# 5. Start the development server
uvicorn main:app --reload
```

> [!TIP]
> On first start, you will see `Building ChromaDB RAG index...` — this is the one-time embedding step and takes ~15-30 seconds. Subsequent restarts are instant.

### Interactive API Docs
Visit **[http://localhost:8000/docs](http://localhost:8000/docs)** to test all endpoints interactively with Swagger UI.

---

## 6. Testing

```bash
# Install test dependencies (once)
pip install pytest httpx

# Run the test suite
python -m pytest test_main.py -v
```

> [!NOTE]
> Use `python -m pytest` instead of `pytest` directly to ensure tests run inside your virtual environment with all installed packages.

### Test Coverage

| Test | What it validates |
|---|---|
| `test_health_endpoint` | API is alive and returns 200 OK |
| `test_intent_clarifying` | Vague prompts return a question, no recommendations |
| `test_intent_ready_to_recommend` | Specific prompts return relevant recommendations from the catalog |
| `test_intent_off_topic` | Guardrail rejects out-of-scope requests |
| `test_conversation_state_accumulation` | Constraints from multiple turns are combined correctly |

---

## 7. File Structure

```
shl_project/
├── main.py            # FastAPI app, schemas, routing logic
├── rag_pipeline.py    # ChromaDB indexing and vector search
├── catalog.json       # Ground-truth SHL test catalog
├── catalog_helper.py  # Utility helpers for catalog operations
├── test_main.py       # Automated test suite (pytest)
├── requirements.txt   # Python dependencies
├── .env               # API key (not committed to version control)
└── chroma_db/         # Persisted vector index (auto-generated, not committed)
```

---

## 8. Known Limitations & Future Work

| Area | Current State | Improvement |
|---|---|---|
| **Filtering** | Semantic search only; no structured filters for duration or language | Add ChromaDB `where` clause for metadata filters |
| **Re-ranking** | Top-K by cosine similarity | Add a cross-encoder re-ranker for more precise ranking |
| **Compare intent** | Uses last message only for comparison query | Extract both assessment names from full history |
| **Stale DB** | Requires manual `rm -rf chroma_db/` if `catalog.json` is updated | Add a checksum or version hash to detect catalog changes at startup |
| **Evaluation** | No quantitative metrics | Add a labelled eval set and measure Precision@K and NDCG |
