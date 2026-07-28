# Evaluation

This is the evaluation write-up for the KhmerSME Knowledge Search RAG system
(see `README.md` for setup and architecture). All numbers below were produced
by running the harness locally, not estimated:

```bash
python run_eval.py --backend st --top-k 3      # primary backend
python run_eval.py --backend tfidf --top-k 3   # baseline for comparison
```

`eval/test_queries.json` holds 10 queries — 8 answerable ones annotated with
the source file(s) they should retrieve, and 2 off-domain queries that must
trigger the refusal gate. Corpus: 22 documents → 44 chunks (chunk_size=120
words). Threshold: 0.30 for sentence-transformers, 0.15 for TF-IDF.

## Results — sentence-transformers (`all-MiniLM-L6-v2`), top-k = 3

| # | Query | Expected source(s) | Top score | Hit@3 | Precision@3 | RR |
|---|---|---|---|---|---|---|
| 1 | How are small and medium enterprises defined in Cambodia? | `01_sme_definition_cambodia` | 0.867 | ✅ | 0.33 | 1.00 |
| 2 | What is the KhmerSME platform and what services does it provide? | `03_khmersme_platform` | 0.850 | ✅ | 0.67 | 1.00 |
| 3 | How does the Credit Guarantee Corporation help businesses that lack collateral? | `06_credit_guarantee`, `05_sme_bank`, `15_women_entrepreneurs` | 0.641 | ✅ | 1.00 | 1.00 |
| 4 | What taxes does a registered small business have to pay in Cambodia? | `07_tax_obligations`, `04_business_registration` | 0.844 | ✅ | 0.67 | 1.00 |
| 5 | How do QR code payments through Bakong benefit market vendors? | `09_bakong_digital_payments` | 0.694 | ✅ | 0.67 | 1.00 |
| 6 | What certifications does a small food producer need to sell to supermarkets? | `14_food_safety_standards`, `13_agro_processing` | 0.694 | ✅ | 1.00 | 1.00 |
| 7 | What challenges do women entrepreneurs face when getting loans? | `15_women_entrepreneurs`, `06_credit_guarantee` | 0.664 | ✅ | 1.00 | 1.00 |
| 8 | Why is building AI chatbots for the Khmer language technically difficult? | `20_ai_for_sme_services` | 0.654 | ✅ | 0.67 | 1.00 |
| 9 | Who won the FIFA World Cup final? | *(none — refusal expected)* | 0.025 | — refused ✅ | — | — |
| 10 | What is the recipe for Italian carbonara pasta? | *(none — refusal expected)* | 0.150 | — refused ✅ | — | — |

**Summary:** hit@3 = 100%, precision@3 = 75%, MRR = 1.00, refusal accuracy = 100% (2/2)

## Results — TF-IDF baseline, top-k = 3

Same aggregate scores — hit@3 = 100%, precision@3 = 75%, MRR = 1.00, refusal
accuracy = 100% — but the raw cosine scores are much lower and closer to the
refusal threshold. Notably, query 4 ("taxes") scored **0.184** against a
**0.15** refusal threshold: a margin of only 0.034. A slightly different
paraphrase of that query could plausibly fall under the threshold and get
incorrectly refused. The sentence-transformers backend has no such near-miss
— its lowest passing score (0.641) sits well above threshold, and its
highest refusal-test score (0.150) sits well below it.

## Discussion

**What worked:**

- **Ranking quality is excellent on both backends.** MRR = 1.00 and
  hit@3 = 100% across all 8 answerable queries — the correct source document
  is always ranked first, even for the TF-IDF baseline. The 22 documents are
  topically distinct enough that lexical overlap alone is a strong enough
  signal for this corpus size.
- **The refusal gate works and, for the primary backend, works with a
  comfortable margin.** Both off-domain queries (World Cup, carbonara recipe)
  scored far below the sentence-transformers threshold (0.025 and 0.150
  against a 0.30 cutoff) and were correctly refused rather than answered
  from the LLM's general knowledge — the core anti-hallucination guarantee
  the project is built around.

**What didn't work / what the numbers actually reveal:**

- **Precision@3 = 75% is a corpus-density ceiling, not a retrieval defect.**
  Checking the chunk store directly: with `chunk_size=120` words and an
  average document length of ~162 words, **every single document splits
  into exactly 2 chunks** (44 chunks / 22 docs, min = max = 2). That means
  for any single-source query, precision@3 can never exceed 2/3 = 0.67 no
  matter how good the embeddings are — the third slot is mathematically
  forced to come from a different document. Query 1 (0.33) and the other
  single-source queries (0.67) hit exactly this ceiling; only the queries
  annotated with multiple acceptable source files (3, 6, 7) reach 1.00,
  which is why the mean lands at 75% instead of near 100%. **The original
  draft of this write-up attributed the gap to "thematically adjacent
  documents" — that guess was wrong; the real cause is chunk count, verified
  by inspecting `build_chunk_records()` output directly.** The actionable
  fix is to evaluate precision@2 (the value chunking actually supports) or
  increase `chunk_size` so single documents can hold 3+ chunks.
- **The TF-IDF refusal threshold has almost no safety margin** (0.184 vs.
  0.15 on the tax query, above). This is a real fragility of the baseline,
  not the primary backend, but it's a concrete illustration of why the
  per-backend threshold was hand-tuned rather than shared.
- **A bug found outside the formal test suite: the conversational-intercept
  regex hijacks legitimate on-topic questions.** `generate.py`'s greeting/
  thanks detector matches on bare substrings like `love` and `thanks`
  anywhere in the query, before retrieval ever runs. Verified directly:

  ```
  Query: "I would love to start an export business, what documents do I need?"
  Response: "Thank you! I am happy to help you with your KhmerSME inquiries!"

  Query: "Thanks in advance, what taxes does a small business pay?"
  Response: "You're very welcome! Let me know if you need information
             regarding business registration, tax laws, or SME support."
  ```

  Both are real, answerable, in-domain questions that get swallowed by a
  canned reply and never reach retrieval or the LLM. None of the 10 queries
  in `eval/test_queries.json` happen to contain a trigger word, so this
  failure mode is invisible to the metrics above — it only surfaces from
  manually probing the code. It should be scoped to require the trigger
  word as the *entire* message (or dropped in favor of just letting the LLM
  handle chit-chat) rather than matching anywhere in longer queries.
- **Evaluation labels are at document level, not chunk level.** A retrieved
  chunk from the right file but the wrong half of it still counts as a hit.
  Given each document is only 2 chunks, this is a minor caveat here, but
  would matter more on longer source documents.

## Suggested next steps

1. Re-run with `--top-k 2` (or raise `chunk_size`) to get a precision metric
   that isn't capped by corpus structure, and re-report both backends.
2. Fix the conversational-intercept regexes to match whole/short messages
   only, then add a couple of adversarial "on-topic sentence containing a
   trigger word" cases to `eval/test_queries.json` so this class of bug is
   caught by `run_eval.py` automatically instead of by hand.
3. Widen the TF-IDF refusal margin (e.g. raise its threshold, or drop it
   from the shipped app now that sentence-transformers is the default and
   keep it only as an offline evaluation baseline).
