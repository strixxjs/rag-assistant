#!/usr/bin/env python3
import json
import urllib.request
import random
import re

QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_eval"

# 1. Exact count
req = urllib.request.Request(
    f"{QDRANT_URL}/collections/{COLLECTION}/points/count",
    data=json.dumps({"exact": True}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
    count_res = json.loads(resp.read().decode("utf-8"))
exact_chunks = count_res["result"]["count"]

# 2. Collect chunks
offset = None
docs = set()
all_chunks = []
table_chunks = []

while True:
    payload_data = {"limit": 1000, "with_payload": True, "with_vector": False}
    if offset:
        payload_data["offset"] = offset
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        data=json.dumps(payload_data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))["result"]
    
    points = res.get("points", [])
    if not points:
        break
    for p in points:
        payload = p.get("payload", {})
        sf = payload.get("source_file", "")
        if sf:
            docs.add(sf)
        all_chunks.append(p)
        text = payload.get("text", "")
        # Look for chunks from docs that have tables (e.g. models/fields, request-response, querysets)
        if any(k in sf for k in ["models/fields", "request-response", "lookups", "settings", "status"]):
            # Check if text contains typical table column contents (like multiple keywords or field options)
            if "Field options" in text or "null" in text and "blank" in text or "status_code" in text:
                table_chunks.append(p)
        
    offset = res.get("next_page_offset")
    if not offset:
        break

print(f"EXACT_CHUNKS: {exact_chunks}", flush=True)
print(f"EXACT_DOCS: {len(docs)}", flush=True)

# 5 random samples (300 chars)
random.seed(42)
sample_points = random.sample(all_chunks, 5)

print("\n=== 5 SAMPLE CHUNKS (300 chars) ===", flush=True)
for i, pt in enumerate(sample_points, 1):
    payload = pt.get("payload", {})
    text = payload.get("text", "")
    repo = payload.get("repo", "")
    sf = payload.get("source_file", "")
    print(f"\n[SAMPLE {i}] ID: {pt['id']} | Repo: {repo} | File: {sf}", flush=True)
    print(text[:300], flush=True)
    print("-" * 60, flush=True)

print("\n=== TABLE CHUNK INSPECTION ===", flush=True)
if table_chunks:
    t_chunk = table_chunks[0]
    payload = t_chunk.get("payload", {})
    print(f"ID: {t_chunk['id']} | File: {payload.get('source_file')}", flush=True)
    print("Cleaned text content:")
    print(payload.get("text", "")[:600], flush=True)
else:
    print("No table chunk candidate found in initial filter, searching...")
