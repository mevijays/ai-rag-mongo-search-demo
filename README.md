AI RAG Mongo + OpenAI Demo

A minimal RAG app using:
- Flask (single-file app with inline UI)
- MongoDB 4.4 (Docker, Apple Silicon friendly)
- OpenAI embeddings + chat
- LangChain text splitters (for chunking)

Upload a PDF/DOCX, generate embeddings, and ask questions over the uploaded knowledge.

**Prerequisites**
- macOS with Docker Desktop
- Python 3.11+
- OpenAI API key

**Quick Start**

1) Configure environment

Create `.env` in the project root. Example:

```
MONGO_INITDB_ROOT_USERNAME=mongo
MONGO_INITDB_ROOT_PASSWORD=mongo
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB_NAME=ragdb
VECTOR_COLLECTION_NAME=documents
OPENAI_API_KEY=sk-...your-key...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
FLASK_HOST=127.0.0.1
FLASK_PORT=8000
FLASK_DEBUG=0
```

2) Start MongoDB + seed sample docs

```zsh
chmod +x setup.sh
./setup.sh
```

3) Create venv, install deps, run app

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
export $(grep -v '^#' .env | xargs)
python app.py
```

Open http://127.0.0.1:8000

**How To Demo (5–7 minutes)**
- RAG = Retrieve relevant chunks + Generate answer with LLM.
- Show app UI:
  - Left: Upload a document (PDF/DOCX)
  - Right: Ask a question (search)
- Upload content:
  - Use the provided sample `demo/sample_faq.md` converted to DOCX/PDF (see below) or your own doc
  - After upload, app chunks text, calls OpenAI embeddings, stores vectors in MongoDB
- Ask a question:
  - e.g., "How do I start the database?" or "Which model generates embeddings?"
  - App embeds the query, finds similar chunks in MongoDB, sends context + question to Chat model, and returns the answer
- Mention fallback: If context doesn’t contain the answer, it falls back to general LLM answer

Preparing the demo document
- A sample is in `demo/sample_faq.md`. Optionally export it as DOCX or PDF:
  - Open it, copy into Word/Pages → Save as DOCX/PDF
  - Or use a markdown-to-pdf tool

Suggested Questions
- What database and vector approach does this app use?
- How do I start the database?
- Which model is used for embeddings?
- What file types can I upload?

Troubleshooting
- Apple Silicon + MongoDB: We use `mongo:4.4` with `platform: linux/amd64` in `docker-compose.yml`.
- If `setup.sh` times out, run `docker logs rag-mongodb` to ensure Mongod is listening.
- If OpenAI client errors about proxies: we pinned `httpx`/`httpcore` in `requirements.txt`.
- Reinstall deps after edits:
  ```zsh
  source .venv/bin/activate
  pip install -r requirements.txt --upgrade
  ```

Tech Notes
- Chunking: LangChain `RecursiveCharacterTextSplitter`
- Embeddings: OpenAI `text-embedding-3-small`
- Chat: OpenAI `gpt-4o-mini`
- Storage: MongoDB collection `documents` with text + embedding per chunk
- Similarity: cosine similarity computed in app over a capped set of docs

Security
- Secrets via `.env` only (do not commit real keys)
- Do not expose this demo publicly without appropriate security
