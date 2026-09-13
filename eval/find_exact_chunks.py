#!/usr/bin/env python3
import json
import urllib.request
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_eval"

def search(query, limit=5, repo=None):
    vec = embedder.encode(query, normalize_embeddings=True).tolist()
    body = {"vector": vec, "limit": limit, "with_payload": True}
    if repo:
        body["filter"] = {"must": [{"key": "repo", "match": {"value": repo}}]}
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())["result"]

topics = {
    "SSL Proxy (Entry 30)": [
        ("SECURE_PROXY_SSL_HEADER HTTP_X_FORWARDED_PROTO reverse proxy", "django"),
        ("SECURE_SSL_REDIRECT https redirect", "django"),
    ],
    "CORS preflight failing": [
        ("CORS preflight OPTIONS request headers allow_origins middleware", "fastapi"),
        ("CORS headers preflight request django-cors-headers", "django"),
        ("CORS preflight OPTIONS method Access-Control-Allow-Origin", "flask")
    ],
    "Cached responses stale after deploy": [
        ("cache versioning prefix stale invalidate after deploy cache.set version", "django"),
        ("cache-control max-age etag stale conditional request", "django")
    ],
    "Pagination duplicates across pages": [
        ("pagination duplicate items order_by inconsistent pagination slice", "django"),
        ("paginator ordering duplicates across pages", "django")
    ],
    "Async route blocked by sync call": [
        ("async def blocking threadpool run_in_threadpool sync function", "fastapi"),
        ("async route blocking event loop async_to_sync sync_to_async", "django")
    ],
    "Migration NOT NULL column existing rows": [
        ("migrations add field NOT NULL default value existing rows", "django"),
        ("migrations.AddField not null default", "django")
    ],
    "Large file upload rejected": [
        ("request body too large 413 Payload Too Large MAX_CONTENT_LENGTH", "flask"),
        ("DATA_UPLOAD_MAX_MEMORY_SIZE file upload maximum size 413", "django"),
        ("UploadFile SpooledTemporaryFile large file memory limit", "fastapi")
    ]
}

print("=== SEARCHING TOPICS IN RAG_EVAL ===")
for title, queries in topics.items():
    print(f"\n==================== {title} ====================")
    for q, repo in queries:
        print(f"\n--- Query: [{repo}] {q} ---")
        hits = search(q, limit=2, repo=repo)
        for h in hits:
            p = h["payload"]
            print(f"ID: {h['id']} | Score: {h['score']:.4f} | Source: {p.get('source_file')}")
            print(p.get("text", "")[:250].replace("\n", " "))
