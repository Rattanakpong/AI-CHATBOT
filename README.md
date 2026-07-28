# KhmerSME Knowledge Search — RAG-Based AI Search System

**CS382 Final Project** · Retrieval-Augmented Generation search over a Cambodian
SME development & digital economy knowledge base (22 documents).

A user asks a question in a Streamlit interface; the system retrieves the most
relevant chunks from the document collection using sentence-transformer
embeddings and cosine similarity, then an LLM generates an answer grounded
**only** in those chunks, with inline citations. Off-topic questions are
refused rather than answered from the model's general knowledge.

Full evaluation write-up (10 test queries, real scores, discussion of what
worked and what didn't): **[EVALUATION.md](EVALUATION.md)**.

---

## 1. Setup & quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) and
embeds the 22-document corpus; embeddings are cached to `.cache/` (keyed by a
hash of the corpus + backend name) so every later start is instant.

### Configuring an LLM provider

Copy `.env.example` to `.env` and fill in **one** provider. The code
auto-detects which one you configured by which env vars are set — no code
change needed to switch:

| Provider | Cost | Env vars |
|---|---|---|
| **Groq** | Free tier, no credit card | `LLM_BASE_URL=https://api.groq.com/openai/v1`, `LLM_API_KEY`, `LLM_MODEL` |
| **Google Gemini** | Free tier | `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`, `LLM_API_KEY`, `LLM_MODEL` |
| **Ollama** | Free, local, offline | Install from ollama.com, `ollama pull llama3.1:8b`, `LLM_BASE_URL=http://localhost:11434/v1` |
| **Anthropic Claude** | Paid | `ANTHROPIC_API_KEY` only |

```bash
# Example: Groq (free)
export LLM_BASE_URL=https://api.groq.com/openai/v1
export LLM_API_KEY=gsk_...
export LLM_MODEL=llama-3.3-70b-versatile
streamlit run app.py
```

Without any provider configured (or if the API call fails), switch
**Answer mode → extractive** in the sidebar: the full retrieval pipeline
still runs and the app shows the raw ranked passages instead of a generated
answer. This also makes the retrieval pipeline testable completely offline.

### Running the evaluation harness

```bash
python run_eval.py                     # sentence-transformers backend (default)
python run_eval.py --backend tfidf     # TF-IDF baseline, for comparison
python run_eval.py --top-k 5           # override retrieval depth
```

See [EVALUATION.md](EVALUATION.md) for the results this produces and what
they mean.

---

## 2. Architecture

```
rag_final_project/
├── app.py                   # Streamlit UI: chat box, answer panel, sources
│                             #   panel, sidebar settings, latency display
├── run_eval.py               # CLI entry point for evaluation
├── requirements.txt
├── .env.example               # provider config template (no real keys)
├── data/sme_docs/            # 22-document knowledge base (Cambodian SME domain)
├── eval/test_queries.json    # 10 test queries (8 answerable + 2 refusal tests)
└── rag/
    ├── ingest.py             # load .txt/.pdf, sentence-aware chunking
    ├── embed_store.py        # pluggable embedding backends, cosine
    │                         #   similarity search, disk-cached embeddings
    ├── generate.py           # grounded LLM answer w/ citations, refusal
    │                         #   gate, input sanitization
    └── evaluate.py           # hit@k, precision@k, MRR, refusal accuracy
```

Pipeline: **Ingest → Chunk → Embed → Vector store → Retrieve → Generate → UI**.
Each stage is a separate module behind a stable interface, so a layer (e.g.
the embedding backend, or the LLM provider) can be swapped without touching
the rest.

### Key design decisions

| Decision | Choice | Justification |
|---|---|---|
| Chunking | Sentence-aware, ~120-word budget, 1-sentence overlap (`rag/ingest.py`) | Chunks never cut a sentence in half, so every retrieved chunk is readable and independently citable; overlap preserves context across boundaries. In practice, with this corpus's average document length (~162 words), this produces exactly 2 chunks per document — see the precision@k caveat in [EVALUATION.md](EVALUATION.md). |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Real semantic embeddings, local, free, no API key. TF-IDF (`rag/embed_store.py::TfidfBackend`) is kept only as an evaluation baseline. |
| Vector search | In-memory cosine similarity (NumPy, `rag/embed_store.py::_cosine_sim_matrix`) | Corpus is ~44 chunks; fine for anything below a few thousand chunks. An `OpenAIBackend` class exists as an optional alternative embeddings API; swapping in FAISS/Chroma would mean replacing the linear scan in `VectorStore.query`. |
| Generation | Provider-agnostic: any OpenAI-compatible endpoint (Groq/Gemini/Ollama) or Anthropic, selected via env vars (`rag/generate.py`) | One strict grounding system prompt for all providers: answer only from numbered `<context>` sources, cite inline as [1][2], answer from general knowledge only for conversational/off-domain input. Avoids lock-in to a single free tier. |
| Graceful failure | Two independent layers | (1) A retrieval-score gate: if the best cosine score is below the backend's threshold (0.30 for sentence-transformers, 0.15 for TF-IDF), the system refuses before calling the LLM. (2) The system prompt also instructs the model not to fabricate citations when context is insufficient. |
| Input handling | Keyword-based prompt-injection filter + greeting/thanks intercept (`rag/generate.py::sanitize_input`, `CONVERSATIONAL_PATTERNS`) | Blocks a short list of known override phrases ("ignore previous instructions", etc.) and short-circuits obvious greetings in English/Khmer before hitting the LLM. **Known false-positive risk** — see Limitations. |
| Caching | Corpus-hash-keyed embedding cache on disk (`.cache/`) | Instant restarts; changing any document, the chunk size, or the backend changes the hash and invalidates the cache automatically. |
| Error handling | Empty-query warning; failed LLM calls surface as an inline error message rather than crashing the app | A dead API key or network outage during a live demo degrades gracefully instead of taking down the UI. |

---

## 3. Dataset

`data/sme_docs/` contains 22 curated `.txt` documents (~162 words each on
average → 44 chunks total, exactly 2 per document at the default chunk size)
covering Cambodian SME development and the digital economy: SME definitions,
MISTI's role, the KhmerSME platform, business registration, SME Bank, credit
guarantees, taxation, the E-Commerce Law, Bakong/KHQR payments, Techo Startup
Center, digital economy policy, industrial policy, agro-processing, food
safety standards, women entrepreneurs, handicrafts, tourism SMEs, export
procedures, the digital skills gap, AI in government services, green
manufacturing, and intellectual property. PDF ingestion is also supported
(`rag/ingest.py::_load_pdf`) if `pypdf` is installed, though the shipped
corpus is all `.txt`.

---

## 4. Known limitations

* **Corpus is English-language.** A production Khmer-language assistant would
  need Khmer-aware embeddings and segmentation — Khmer script has no word
  spaces, so the current regex sentence splitter would not work on it.
* **In-memory search won't scale** past a few thousand chunks — swap in FAISS
  or Chroma via the backend interface in `embed_store.py` (an `OpenAIBackend`
  is already stubbed as an example of adding a backend).
* **The sentence splitter is regex-based** (`re.compile(r"(?<=[.!?])\s+")`);
  abbreviations like "e.g." or "U.S." can cause occasional over-splitting.
* **Precision@k is structurally capped by corpus density**, not a retrieval
  quality signal, at the default chunk size — every document in this corpus
  splits into exactly 2 chunks, so precision@3 tops out at 2/3 for
  single-source queries regardless of embedding quality. See
  [EVALUATION.md](EVALUATION.md) for the measured impact.
* **Evaluation labels are document-level, not chunk-level** — a retrieved
  chunk from the right file but the wrong paragraph still counts as a hit.
* **The conversational-intercept regex has a real false-positive bug.**
  `CONVERSATIONAL_PATTERNS` in `rag/generate.py` matches bare words like
  `love`, `thanks`/`thank you`, and `hi`/`hello` *anywhere* in the query, not
  just as full greetings — so an on-topic question like "I would love to
  start an export business, what documents do I need?" gets swallowed by a
  canned reply and never reaches retrieval. Confirmed by direct testing; see
  [EVALUATION.md](EVALUATION.md) for the reproduction. Not currently caught
  by `eval/test_queries.json` since none of those queries contain a trigger
  word.
* **The `sanitize_input` prompt-injection filter is a short hardcoded phrase
  list**, not a robust defense — it only catches the exact phrases listed
  (`"ignore previous instructions"`, etc.) and can be trivially bypassed by
  rewording.
* **Free LLM API tiers (Groq, Gemini) are rate-limited and can change without
  notice**; local Ollama or extractive mode are the dependable fallbacks for
  a live demo.
* **TF-IDF's refusal threshold has a thin safety margin** on at least one
  in-domain query in the eval set (0.184 vs. a 0.15 threshold) — a
  reasonable paraphrase could tip it into an incorrect refusal. The
  sentence-transformers backend does not show this problem (see
  [EVALUATION.md](EVALUATION.md)).

---

## 5. Requirements mapping

1. Ingestion: 22 docs, `rag/ingest.py` (+ optional PDF via `pypdf`)
2. Chunking: sentence-aware with overlap
3. Real embeddings: sentence-transformers (TF-IDF kept only as a baseline)
4. Vector search: in-memory cosine top-k
5. Generation: cited, grounded LLM answers (provider-agnostic)
6. Graceful failure: two-layer refusal gate
7. Interface: Streamlit with retrieval + answer + sources panels, sidebar
   settings, and latency display
8. Evaluation: 10 queries, 4 metrics, write-up — see [EVALUATION.md](EVALUATION.md)
9. Documentation: this README + EVALUATION.md
