#!/usr/bin/env python3
import json
import urllib.request
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_eval"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "

def search(query, limit=5, repo=None, use_prefix=False):
    q_str = (BGE_PREFIX + query) if use_prefix else query
    vec = embedder.encode(q_str, normalize_embeddings=True).tolist()
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

# ---------------------------------------------------------------------------
# CHECK 2: Collection configuration
# ---------------------------------------------------------------------------
req = urllib.request.Request(f"{QDRANT_URL}/collections/{COLLECTION}")
with urllib.request.urlopen(req) as resp:
    col_info = json.loads(resp.read().decode())["result"]

print("=== CHECK 2: COLLECTION CONFIG ===")
print("Collection Status:", col_info.get("status"))
print("Vectors Config:", col_info.get("config", {}).get("params", {}).get("vectors"))
print()

# ---------------------------------------------------------------------------
# CHECK 3: Manual Sanity Probes (With and Without Prefix)
# ---------------------------------------------------------------------------
print("=== CHECK 3: SANITY PROBES ===")
probes = [
    ("select_related", "django"),
    ("connection pooling keep-alive", "httpx"),
    ("My template loop is firing hundreds of identical database queries, one per item. How do I fetch the related objects in a single query?", None),
    ("How can I avoid the latency of opening a new TCP connection and SSL handshake on every outbound HTTP request in my microservice?", None),
]

for p_text, repo in probes:
    print(f"\n--- Probe: '{p_text[:70]}...' (Repo filter: {repo}) ---")
    
    hits_no_prefix = search(p_text, limit=3, repo=repo, use_prefix=False)
    print("  [Without prefix]:")
    for h in hits_no_prefix:
        print(f"    ID: {h['id']} | Score: {h['score']:.4f} | Source: {h['payload'].get('source_file')}")
        print(f"    Text: {h['payload'].get('text', '')[:120].replace(chr(10), ' ')}")

    hits_with_prefix = search(p_text, limit=3, repo=repo, use_prefix=True)
    print("  [WITH bge prefix]:")
    for h in hits_with_prefix:
        print(f"    ID: {h['id']} | Score: {h['score']:.4f} | Source: {h['payload'].get('source_file')}")
        print(f"    Text: {h['payload'].get('text', '')[:120].replace(chr(10), ' ')}")

# ---------------------------------------------------------------------------
# CHECK 4: Full Chunk Text for Entries 07 and 09
# ---------------------------------------------------------------------------
print("\n=== CHECK 4: FULL CHUNK TEXT FOR ENTRIES 07 & 09 ===")
with open("eval/dataset_with_meta.json") as f:
    entries = json.load(f)["entries"]

e07 = entries[6]  # 0-indexed #7
e09 = entries[8]  # 0-indexed #9

for e, num in [(e07, "07"), (e09, "09")]:
    cid = e["ground_truth_chunk_id"]
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        data=json.dumps({"ids": [cid], "with_payload": True}).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        pts = json.loads(resp.read().decode())["result"]
    
    print(f"\n--- Entry {num} Ground Truth Chunk ---")
    print(f"ID: {cid}")
    print(f"Question: {e['question']}")
    print(f"Source file: {e.get('_source_file')}")
    if pts:
        p = pts[0]["payload"]
        print(f"Chunk Index: {p.get('chunk_index')}")
        print("FULL TEXT:")
        print(p.get("text", ""))
    else:
        print("POINT NOT FOUND IN QDRANT!")
