# Evidence-Based Fake News Detection Using NLP and Retrieval-Augmented Generation

## 1. Abstract
Most fake-news detectors are binary classifiers trained on writing style, which generalise poorly and give no
justification. This project instead treats misinformation detection as **evidence-based claim verification**:
for each factual claim the system retrieves web evidence, ranks it with classical NLP, and uses a Retrieval-Augmented
Generation (RAG) step to produce a **cited, explainable verdict** (*Supported*, *Contradicted*, *Insufficient Evidence*).
The design is deliberately lightweight (TF-IDF, regex NLP, hosted LLM API) so it runs on a laptop with no GPU.

## 2. Problem statement and objectives
- Verify claims against external evidence rather than predicting "fake/real" from style.
- Provide transparent reasoning and source links for every verdict.
- Represent uncertainty explicitly with an *Insufficient Evidence* class.
- Keep resource usage low (no deep-learning frameworks or model downloads).

## 3. System architecture
```
 User input (claim / headline / article / URL)
        │
        ▼
 [1] Claim extraction ──► check-worthy claims C1..Cn
        │            (per claim)
        ▼
 [2] Evidence retrieval: Tavily search → fetch pages (Requests + BeautifulSoup)
        │
        ▼
 [3] Passage ranking: sentence windows → TF-IDF cosine + key-term coverage → top-k
        │
        ▼
 [4] RAG verdict: Groq LLM (JSON, evidence ids) → [5] citation validation / guardrails
        │
        ▼
 [6] Aggregation (overall verdict) → SQLite history → Streamlit explanation UI
```

## 4. Methodology
### 4.1 Claim extraction (`claim_extractor.py`)
Sentences are split with a regex splitter that protects abbreviations and decimals. Each sentence receives a
check-worthiness score: `+2` digits, `+1.5` date/month tokens, `+1` per named entity (max 3), `+1.5` factual/reporting
verb, `+0.5` intensifier (first, never, million); `−1.5` per opinion marker, `−3` question, length penalties.
Top sentences above a threshold (default 2.0) are kept, near-duplicates removed (Jaccard > 0.8), original order restored.
A single sentence/headline is verified as-is.

### 4.2 Evidence retrieval (`retriever.py`)
Tavily returns candidate URLs and query-focused snippets. Pages are downloaded in parallel, boilerplate tags removed,
and `<p>/<li>` text from `<article>/<main>` is kept. If a page cannot be fetched (paywall, bot-block), the Tavily snippet is used.

### 4.3 Semantic similarity (`similarity.py`)
Pages are split into overlapping 3-sentence windows. A `TfidfVectorizer` (English stop-words, uni+bi-grams, sublinear TF)
is fitted on the claim plus all passages. The final score is

`score = 0.75 · cos(claim, passage) + 0.25 · coverage(claim key terms in passage)`

where key terms are numbers and capitalised tokens. The coverage term penalises passages that discuss the topic but
mismatch the specific number/name. At most two passages per page are kept for source diversity.

### 4.4 RAG verdict (`verdict.py`)
The top passages (labelled E1…En with domain and reliability label) are inserted into the prompt. The model is
instructed to use only the evidence, treat page text as untrusted (prompt-injection defence), cite ids, and return JSON.
Temperature is 0.1 for reproducibility.

### 4.5 Guardrails
Cited ids not present in the retrieved set are discarded. *Supported* or *Contradicted* without a valid citation is
downgraded to *Insufficient Evidence* with capped confidence. Confidence is clamped to [0, 1]. Malformed JSON is
recovered via fenced/embedded-JSON parsing, otherwise reported as an error.

### 4.6 Aggregation
Any *Contradicted* claim → overall *Contradicted*; all *Supported* → *Supported*; otherwise *Insufficient Evidence*.
Mixed submissions are flagged in the summary. Failures for one claim do not abort the others.

### 4.7 Source credibility prior (`credibility.py`)
A transparent domain-list heuristic (fact-checkers 0.95, institutions/.gov/.edu 0.90, major news 0.80, encyclopedia 0.60,
unrated 0.50, social media 0.30). It is displayed and given to the LLM as context; it never decides the verdict by itself.

## 5. Implementation summary
Python 3.11, Streamlit, Groq (OpenAI-compatible REST), Tavily REST, Requests, BeautifulSoup (`html.parser`),
scikit-learn, SQLite (`sqlite3`). Direct REST calls avoid extra SDK dependencies. Dependencies are injected
(`search_fn`, `llm_fn`, HTTP `session`) which makes the whole pipeline testable offline.

## 6. Testing
41 unit tests cover sentence splitting, claim scoring, TF-IDF ranking (including number-mismatch cases), HTML extraction,
HTTP error handling, retry logic, JSON parsing, citation guardrails, aggregation rules, partial-failure isolation and SQLite CRUD.
`scripts/evaluate.py` runs the live pipeline on `data/sample_claims.json` and reports accuracy, a confusion matrix and
per-class precision / recall / F1.

## 7. Evaluation protocol (to fill in with your results)
Run `python scripts/evaluate.py`. Report accuracy and per-class F1. Suggested ablations: (a) without TF-IDF ranking
(raw search snippets only), (b) without the key-term coverage term, (c) without citation validation, (d) different Groq models.
Extend the dataset with claims from public fact-check archives (Snopes, PolitiFact, BOOM, Alt News).

## 8. Limitations
Lexical (not semantic) matching; dependence on search-engine coverage and on web truthfulness; heuristic source ratings;
English-only claim extraction; LLM errors; paywalled pages; time-sensitive claims may match outdated pages.

## 9. Future work
Multilingual support, small local stance/NLI model, cross-encoder re-ranking (optional heavy mode), result caching,
temporal evidence weighting, a larger benchmark (FEVER, LIAR, CheckThat!), and user feedback loops.

## 10. Viva question bank
- *Why not a classifier?* Style-based classifiers do not verify facts, generalise poorly and cannot explain.
- *Why TF-IDF instead of embeddings?* Lightweight, no downloads, interpretable; trade-off is no paraphrase understanding.
- *How do you reduce hallucination?* Evidence-only prompt, JSON schema, citation validation, downgrade rule, low temperature.
- *What is prompt injection here?* Malicious web text instructing the LLM; mitigated by explicit untrusted-content rule.
- *Why an Insufficient Evidence class?* Absence of evidence is not evidence of falsity.
