#!/usr/bin/env python3
import json
import logging
import os
import random
import re
import sys
import time
import urllib.request
import uuid
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_eval"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

log.info("Loading embedder for search...")
embedder = SentenceTransformer(EMBED_MODEL)

def search_qdrant(query: str, limit: int = 5, filter_repo: str = None):
    q_vec = embedder.encode(query, normalize_embeddings=True).tolist()
    body = {
        "vector": q_vec,
        "limit": limit,
        "with_payload": True
    }
    if filter_repo:
        body["filter"] = {"must": [{"key": "repo", "match": {"value": filter_repo}}]}
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())["result"]

def groq_call(prompt: str, max_tokens: int = 400, retries: int = 2) -> str | None:
    import httpx
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.4,
    }
    for attempt in range(retries + 1):
        try:
            r = httpx.post(GROQ_ENDPOINT, headers=headers, json=body, timeout=30)
            if r.status_code == 429:
                log.warning(f"Rate-limited (429), waiting 30s...")
                time.sleep(30)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            log.warning(f"Groq error: {exc}")
            time.sleep(3)
    return None

def parse_qa(raw: str) -> dict | None:
    if not raw or "QUESTION:" not in raw or "ANSWER:" not in raw:
        return None
    try:
        q = raw.split("QUESTION:")[1].split("ANSWER:")[0].strip().strip('"*`')
        a = raw.split("ANSWER:")[1].strip().strip('"*`')
        if len(q) < 15 or len(a) < 10:
            return None
        return {"question": q, "answer": a}
    except Exception:
        return None

# ---------------------------------------------------------------------------
# 1. Ten Hand-written Developer Symptom Questions (Slots 20-29)
# ---------------------------------------------------------------------------
HAND_WRITTEN_QUERIES = [
    {
        "search_query": "select_related prefetch_related queries in template loop",
        "question": "My template loop is firing hundreds of identical database queries, one per item. How do I fetch the related objects in a single query?",
        "answer": "Use select_related() for foreign-key/one-to-one relations to perform a SQL JOIN in a single query, or prefetch_related() for many-to-many/many-to-one relations to batch queries in Python.",
        "repo": "django"
    },
    {
        "search_query": "FastAPI database session dependency injection share connection",
        "question": "My API endpoint opens a brand new database connection on every incoming request instead of reusing one — how do I share a single session per request?",
        "answer": "Declare a database session dependency function using yield and inject it into your path operations via Depends() to ensure clean lifecycle management per request.",
        "repo": "fastapi"
    },
    {
        "search_query": "FastAPI BackgroundTasks return response immediately process data",
        "question": "How do I trigger an email or long-running notification after sending the HTTP response so the client does not have to wait?",
        "answer": "Use FastAPI's BackgroundTasks parameter and call background_tasks.add_task(func, *args) inside your route function to execute tasks right after returning the response.",
        "repo": "fastapi"
    },
    {
        "search_query": "Django transaction.atomic rollback on exception",
        "question": "Why is my database transaction committing changes even when an error occurs during processing, and how do I ensure everything rolls back together?",
        "answer": "Wrap the operations inside a transaction.atomic() context block, which guarantees that all database writes are rolled back if an unhandled exception is raised.",
        "repo": "django"
    },
    {
        "search_query": "Django streaming responses memory usage big file",
        "question": "Serving large file downloads causes my web server to run out of memory. How do I stream files in small chunks without buffering the whole file in RAM?",
        "answer": "Use FileResponse or StreamingHttpResponse with an iterator/generator so Django streams content in chunks without loading the entire payload into memory.",
        "repo": "django"
    },
    {
        "search_query": "Django password storage user model clear text hash",
        "question": "Why shouldn't I assign a plain-text string directly to the user's password attribute when creating or updating user accounts?",
        "answer": "Django requires passwords to be salted and hashed via user.set_password() rather than saved as raw text on the model attribute.",
        "repo": "django"
    },
    {
        "search_query": "Flask working outside of application context error current_app",
        "question": "My background worker fails with an error saying it is working outside of application context when accessing configuration. How do I fix this?",
        "answer": "Push an explicit application context using with app.app_context(): around the code that needs access to current_app or extensions.",
        "repo": "flask"
    },
    {
        "search_query": "httpx Client session connection pooling reuse TCP",
        "question": "How can I avoid the latency of opening a new TCP connection and SSL handshake on every outbound HTTP request in my microservice?",
        "answer": "Instantiate a single httpx.Client or httpx.AsyncClient as a shared context manager or singleton to enable HTTP connection pooling and keep-alive.",
        "repo": "httpx"
    },
    {
        "search_query": "FastAPI validation error 422 Unprocessable Entity path parameters",
        "question": "Why does my endpoint return HTTP status 422 instead of 400 when a client sends invalid types in URL parameters?",
        "answer": "FastAPI uses Pydantic for request validation and automatically generates a 422 Unprocessable Entity response with specific error details when parameter conversion fails.",
        "repo": "fastapi"
    },
    {
        "search_query": "Django lazy translation plural ngettext_lazy",
        "question": "When defining translated model fields with plural forms, why does standard translation fail before the actual count is known?",
        "answer": "Use ngettext_lazy (or npgettext_lazy) to defer evaluation until the plural count is provided when formatting the string.",
        "repo": "django"
    },
]

# ---------------------------------------------------------------------------
# Build the Dataset
# ---------------------------------------------------------------------------
def main():
    entries = []

    # 1. Fetch 10 Hand-written chunk IDs
    log.info("Matching 10 hand-written questions to rag_eval chunks...")
    for hw in HAND_WRITTEN_QUERIES:
        res = search_qdrant(hw["search_query"], limit=3, filter_repo=hw["repo"])
        if not res:
            log.warning(f"No match for query: {hw['search_query']}")
            continue
        best = res[0]
        log.info(f"Matched '{hw['question'][:50]}...' -> Chunk {best['id']} ({best['payload'].get('source_file')})")
        entries.append({
            "id": str(uuid.uuid4()),
            "question": hw["question"],
            "ground_truth_chunk_id": str(best["id"]),
            "ground_truth_answer": hw["answer"],
            "_source_file": best["payload"].get("source_file"),
            "_repo": best["payload"].get("repo"),
        })

    # 2. Scroll chunks for Groq generation (15 single + 5 multi)
    log.info("Fetching pool of chunks for Groq generation...")
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        data=json.dumps({"limit": 500, "with_payload": True, "with_vector": False}).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        all_pts = json.loads(resp.read().decode())["result"]["points"]

    random.seed(42)
    random.shuffle(all_pts)

    SINGLE_PROMPT = """\
You are writing questions for a developer Q&A benchmark testing a search/RAG system.
A developer is frustrated, encountering an issue or seeking a solution. Write ONE question
they might ask — describing a practical SYMPTOM or GOAL in their own words.

Strict Rules:
- Frame as a problem, symptom or goal (e.g., "Why does my request fail when...", "How do I prevent...", "My app keeps...").
- NEVER write a generic doc lookup (NO "What is X", "How does Y work").
- Do NOT reuse words from document headings, titles, or exact API class/method names in the chunk.
- The answer must be found in the chunk.
- Write a SHORT FACTUAL ANSWER (1-2 sentences). Do NOT quote chunk text verbatim.

Chunk (repo: {repo}, file: {source_file}):
\"\"\"
{text}
\"\"\"

Format:
QUESTION: <question describing symptom or goal>
ANSWER: <short factual answer in 1-2 sentences>"""

    MULTI_PROMPT = """\
You are writing questions for a developer Q&A benchmark testing a search/RAG system.
Write ONE question describing a complex problem, symptom, or goal that REQUIRES synthesizing
information from BOTH excerpts below.

Strict Rules:
- Frame as a practical developer symptom or architectural goal.
- Do NOT reuse document headings, titles, or exact API class/method names from either excerpt.
- The answer MUST require facts from BOTH excerpts.
- Write a SHORT FACTUAL ANSWER (2-3 sentences) synthesizing both excerpts. Do NOT quote verbatim.

Excerpt 1 (repo: {repo}, file: {source_file1}):
\"\"\"
{text1}
\"\"\"

Excerpt 2 (same repo, file: {source_file2}):
\"\"\"
{text2}
\"\"\"

Format:
QUESTION: <question synthesizing both excerpts>
ANSWER: <short factual answer in 2-3 sentences>"""

    # Generate 15 single-chunk questions
    log.info("Generating 15 single-chunk questions via Groq...")
    single_generated = 0
    candidate_chunks = [p for p in all_pts if len(p.get("payload", {}).get("text", "")) >= 250]
    
    for pt in candidate_chunks:
        if single_generated >= 15:
            break
        p = pt["payload"]
        text = p.get("text", "")
        repo = p.get("repo", "")
        sf = p.get("source_file", "")
        
        raw = groq_call(SINGLE_PROMPT.format(repo=repo, source_file=sf, text=text[:700]))
        qa = parse_qa(raw)
        if not qa:
            continue
        
        entries.append({
            "id": str(uuid.uuid4()),
            "question": qa["question"],
            "ground_truth_chunk_id": str(pt["id"]),
            "ground_truth_answer": qa["answer"],
            "_source_file": sf,
            "_repo": repo,
        })
        single_generated += 1
        log.info(f"[{single_generated}/15 single] Q: {qa['question']}")
        time.sleep(1.5)

    # Group by repo for multi-chunk
    by_repo = {}
    for pt in all_pts:
        r = pt.get("payload", {}).get("repo", "unknown")
        by_repo.setdefault(r, []).append(pt)

    log.info("Generating 5 multi-chunk questions via Groq...")
    multi_generated = 0
    for repo, r_pts in by_repo.items():
        if multi_generated >= 5:
            break
        long_pts = [p for p in r_pts if len(p.get("payload", {}).get("text", "")) >= 200]
        if len(long_pts) < 2:
            continue
        random.shuffle(long_pts)
        c1, c2 = long_pts[0], long_pts[1]
        
        raw = groq_call(MULTI_PROMPT.format(
            repo=repo,
            source_file1=c1.get("payload", {}).get("source_file", ""),
            source_file2=c2.get("payload", {}).get("source_file", ""),
            text1=c1.get("payload", {}).get("text", "")[:500],
            text2=c2.get("payload", {}).get("text", "")[:500],
        ))
        qa = parse_qa(raw)
        if not qa:
            continue
        
        entries.append({
            "id": str(uuid.uuid4()),
            "question": qa["question"],
            "ground_truth_chunk_id": [str(c1["id"]), str(c2["id"])],
            "ground_truth_answer": qa["answer"],
            "_source_file": f"{c1.get('payload', {}).get('source_file')} + {c2.get('payload', {}).get('source_file')}",
            "_repo": repo,
        })
        multi_generated += 1
        log.info(f"[{multi_generated}/5 multi] Q: {qa['question']}")
        time.sleep(1.5)

    # Save to eval/dataset.json (clean metadata fields before saving)
    clean_entries = []
    for e in entries:
        clean_entries.append({
            "id": e["id"],
            "question": e["question"],
            "ground_truth_chunk_id": e["ground_truth_chunk_id"],
            "ground_truth_answer": e["ground_truth_answer"]
        })
    
    with open("eval/dataset.json", "w", encoding="utf-8") as f:
        json.dump({"entries": clean_entries}, f, indent=2, ensure_ascii=False)
    
    with open("eval/dataset_with_meta.json", "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, indent=2, ensure_ascii=False)

    log.info(f"Saved {len(clean_entries)} items to eval/dataset.json")

if __name__ == "__main__":
    main()
