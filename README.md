# 🔎 Evidence-Based Fake News Detection Using NLP and RAG

An AI-powered fact-checking app that does **not** just label news "fake" or "real".
It extracts factual claims, searches the web for evidence, ranks that evidence,
and uses an LLM (Retrieval-Augmented Generation) to decide whether each claim is
**Supported**, **Contradicted** or has **Insufficient Evidence** — with
reasoning and clickable source citations.

Runs fully on your Mac with **no PyTorch / Transformers / FAISS / model downloads**
(total install is roughly 150 MB, mostly scikit-learn and Streamlit).

---

## ✨ Features

| Component | How it is implemented |
|---|---|
| Claim extraction | Regex sentence splitter + check-worthiness scoring (numbers, dates, entities, factual verbs; penalises opinions/questions) |
| Evidence retrieval | Tavily Search API + `requests` / `BeautifulSoup` page extraction (parallel, with snippet fallback) |
| Semantic similarity | TF-IDF (1–2 grams) + cosine similarity + number/entity coverage, over overlapping sentence windows |
| RAG | Top passages → Groq LLM (JSON output, cites evidence ids) |
| Guardrails | Citations are validated; verdicts without valid citations are downgraded to *Insufficient Evidence* |
| Explainability | Reasoning, supporting / contradicting passages, similarity scores, source-reliability labels |
| History | SQLite (`data/history.db`), per-user history, reopen / delete / clear, JSON report download |
| Authentication | Email/password registration, bcrypt hashing, session tokens (30-day expiry), account settings |
| Input types | Claim, headline, full article text, URL, or **image upload (PNG/JPG/JPEG)** |

## 🗂️ Folder structure

```
fakenews-rag/
├── app.py                  # Streamlit frontend
├── requirements.txt
├── .env.example            # copy to .env and add keys
├── setup_mac.sh            # optional one-command setup
├── src/
│   ├── config.py           # settings / env loading
│   ├── models.py           # dataclasses (Claim, Passage, ClaimResult, ...)
│   ├── claim_extractor.py  # NLP: sentence split + claim scoring
│   ├── retriever.py        # Tavily search + page scraping
│   ├── credibility.py      # domain reliability heuristic
│   ├── similarity.py       # TF-IDF + cosine evidence ranking
│   ├── verdict.py          # Groq RAG prompt, parsing, guardrails
│   ├── pipeline.py         # orchestration + aggregation
│   ├── database.py         # SQLite history (per-user)
│   ├── auth.py             # User auth: register, login, sessions, bcrypt
│   ├── ocr.py              # Tesseract OCR for image text extraction
│   ├── errors.py, utils.py
├── tests/                  # 41+ offline unit tests (no API keys needed)
├── scripts/evaluate.py     # accuracy / confusion matrix on labelled claims
├── data/sample_claims.json # 13 labelled claims for evaluation
└── docs/PROJECT_REPORT.md  # academic write-up (architecture, method, limits)
```

## 🍎 Setup on macOS (beginner friendly)

### 1. Install Python 3.11
Open **Terminal** (Cmd + Space → "Terminal") and run:
```bash
# Install Homebrew first if you don't have it: https://brew.sh
brew install python@3.11
python3.11 --version     # should print Python 3.11.x
```

### 2. Get free API keys
- **Groq**: sign up at https://console.groq.com/keys → *Create API Key*
- **Tavily**: sign up at https://app.tavily.com → copy your API key

### 3. Open the project in VS Code
Unzip the project, then in Terminal:
```bash
cd ~/Downloads/fakenews-rag      # adjust to where you unzipped it
code .                           # opens VS Code (or use File → Open Folder)
```
Use VS Code's built-in terminal (**Terminal → New Terminal**) for the next steps.

### 4. Create a virtual environment and install dependencies
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
*(Shortcut: `bash setup_mac.sh` does steps 4 and the `.env` copy for you.)*

If VS Code asks to select an interpreter, choose `.venv` (Cmd+Shift+P → "Python: Select Interpreter").

### 5. Add your API keys
```bash
cp .env.example .env
```
Open `.env` in VS Code and paste your keys:
```
GROQ_API_KEY=gsk_xxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxx
```

### 6. Run the app
```bash
streamlit run app.py
```
Your browser opens at http://localhost:8501. Try one of the example buttons, then press **Verify**.
Stop the app with `Ctrl + C`. Next time, just run `source .venv/bin/activate` then `streamlit run app.py`.

## 🧪 Running the tests
No API keys or internet needed (HTTP calls are faked):
```bash
python -m pytest -q          # or: python -m unittest discover -s tests -t .
```

## 📊 Evaluating accuracy (for your report)
```bash
python scripts/evaluate.py --limit 13
```
Prints accuracy, a confusion matrix and per-class precision/recall/F1, and saves
`data/evaluation_results.csv`. Add your own labelled claims to `data/sample_claims.json`.

## 🔐 User Authentication

The app includes a complete authentication system:

- **Registration**: Email, username (min 3 chars), password (min 8 chars)
- **Login**: Email + password with bcrypt verification
- **Sessions**: Secure tokens with 30-day expiry, stored in SQLite
- **Per-user history**: Each user only sees their own verification history
- **Account settings**: Change username, change password, delete account
- **Logout everywhere**: Invalidate all sessions on password change

### How it works

1. On first visit, you'll see a **Sign In** screen
2. Click **"Create an account"** to register
3. After login, your verification history is saved to your account
4. Access account settings from the sidebar (username, password, delete account)

## 🖼️ Image Verification

The **Image Verify** tab lets you upload news screenshots, social media posts, newspaper clippings, or headline images.

1. **Upload** a PNG, JPG, or JPEG image
2. **Select OCR languages** (English, Hindi, Marathi — requires corresponding Tesseract language packs)
3. **Extract text** — the app runs Tesseract OCR and shows the extracted text
4. **Edit** the extracted text if needed (OCR isn't perfect)
5. **Verify** — the text goes through the same verification pipeline as text input

### Installing Tesseract language packs

**macOS (Homebrew):**
```bash
brew install tesseract tesseract-lang
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-mar
```

**Windows:**
Download the installer from https://github.com/UB-Mannheim/tesseract/wiki and select "Additional language data" for Hindi and Marathi during installation.

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `zsh: command not found: python3.11` | `brew install python@3.11`, then reopen Terminal |
| "Missing API key(s)" in the app | Check `.env` is in the project root (not `.env.example`) and restart Streamlit |
| "Groq rejected the API key" / "Tavily rejected…" | Re-copy the key; no quotes or spaces |
| "rate limit reached" | Free tiers are limited; wait ~1 minute or lower *Max claims* in the sidebar |
| Many claims come back *Insufficient Evidence* | Rephrase as a plain, specific factual sentence; increase *Web pages per claim* |
| `ModuleNotFoundError` | Activate the venv: `source .venv/bin/activate` |
| "Tesseract not found" / OCR fails | Install Tesseract: `brew install tesseract` (macOS), `sudo apt-get install tesseract-ocr` (Ubuntu), or download from GitHub (Windows). For Hindi/Marathi, install `tesseract-ocr-hin` and `tesseract-ocr-mar`. |
| "bcrypt not found" / Auth fails | Install bcrypt: `pip install bcrypt` (already in requirements.txt) |
| Groq model not found | Models get retired; set `GROQ_MODEL` in `.env` to a current one from console.groq.com/docs/models |

## ⚠️ Limitations
- TF-IDF matches **words, not meaning**: paraphrases can be ranked lower. (Trade-off for staying lightweight.)
- The web can itself contain misinformation; the source-reliability heuristic is a simple domain list.
- LLMs can make mistakes; the app mitigates this with citation validation but cannot remove it entirely.
- Very recent events may have little indexed evidence; paywalled pages fall back to search snippets.
- This is an educational tool, not a replacement for professional fact-checkers.

## 🚀 Future work
Multilingual claims (Hindi/Marathi), stance classification with a small local model, caching of search results,
time-aware evidence weighting, and a larger benchmark (e.g. FEVER / LIAR style data).
