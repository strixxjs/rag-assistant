#!/usr/bin/env python3
import json
import urllib.request
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_eval"

def search(query, limit=2, repo=None):
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

candidates = [
    ("FastAPI sub-dependencies hierarchical parameters", "fastapi"),
    ("FastAPI OAuth2PasswordBearer security token header", "fastapi"),
    ("Django F expressions update field without race condition database level", "django"),
]

for q, repo in candidates:
    print(f"\n--- Query: [{repo}] {q} ---")
    hits = search(q, limit=2, repo=repo)
    for h in hits:
        p = h["payload"]
        print(f"ID: {h['id']} | Score: {h['score']:.4f} | Source: {p.get('source_file')}")
        print(p.get("text", "")[:250].replace("\n", " "))
