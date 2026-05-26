# RAG Assistant

Internal AI knowledge base for company.

## Documentation

[📄 Full solution and architecture description (Google Doc)](https://docs.google.com/document/d/18CLz1yrpGh2dcESVQ0FQ_i0OEqhckRi6CTTyEtRa8ys/edit?usp=sharing)

## Overview

RAG Assistant collects knowledge from Google Docs, Confluence, Slack and GitHub,
indexes it via embeddings and allows anyone to get an answer with a reference to the source
via Slack bot, REST API or Web UI.

## Architecture

![Architecture diagram](architecture2.png)

## Project Structure

- `ingestion.py` — document chunking, embeddings creation and indexing to Qdrant
- `rag.py` — semantic search and answer generation with source references
- `webhook.py` — FastAPI server that receives Google Drive webhooks and triggers reindexing

## Tech Stack

- **Vector DB:** Qdrant
- **Embeddings:** OpenAI `text-embedding-3-small`
- **LLM:** GPT-4o
- **Framework:** FastAPI, LangChain
- **Phase 1 sources:** Google Docs, Confluence
- **Phase 2 sources:** Slack, GitHub

## How It Works

1. Document is loaded → split into chunks → converted to embeddings → stored in Qdrant
2. When a document changes → webhook triggers automatic reindexing
3. User asks a question → semantic search finds relevant chunks → LLM generates answer with source reference

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/strixxjs/rag-assistant
cd allright-brain

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Fill in .env
cp .env.example .env
# Open .env and add your API keys

# 5. Run Qdrant in Docker
docker run -p 6333:6333 qdrant/qdrant

# 6. Start the webhook server
uvicorn webhook:app --reload
```

Swagger UI available at `http://127.0.0.1:8000/docs`.

## Current Limitations

This is a concept, not production-ready code. For production use:

- Move client creation (`QdrantClient`, `OpenAIEmbeddings`, `ChatOpenAI`) to singletons or FastAPI `Depends`
- Add retry with exponential backoff for OpenAI and Qdrant calls (e.g. via `tenacity`)
- Replace `uuid.uuid4()` with deterministic id (`hash(source_url + chunk_index)`) to avoid duplicates on reindexing
- Add logic to delete old chunks by `source_url` before re-upsert
- Replace `datetime.utcnow()` with `datetime.now(timezone.utc)` (deprecated in Python 3.12)
- Create Qdrant collection with correct `vector_size=1536` via `get_or_create_collection`
- Add authentication for webhook endpoint
- Cover critical paths with tests (ingestion, retrieval)

## Note

This is a concept implementation, not production-ready code.
