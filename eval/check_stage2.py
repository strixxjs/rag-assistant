#!/usr/bin/env python3
import json
import urllib.request
import random

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

# 2. Get all distinct doc files by scrolling through payloads
offset = None
docs = set()
all_samples = []

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
        sf = payload.get("source_file") or payload.get("metadata", {}).get("source_file") or payload.get("source")
        if sf:
            docs.add(sf)
        all_samples.append(p)
        
    offset = res.get("next_page_offset")
    if not offset:
        break

print(f"EXACT_CHUNK_COUNT: {exact_chunks}")
print(f"EXACT_DOC_COUNT: {len(docs)}")

# Pick 5 random chunks (especially including django/rst)
django_points = [p for p in all_samples if "django" in json.dumps(p.get("payload", {}))]
fastapi_points = [p for p in all_samples if "fastapi" in json.dumps(p.get("payload", {}))]

sample_points = random.sample(django_points, min(3, len(django_points))) + random.sample(all_samples, 2)
sample_points = sample_points[:5]

print("--- 5 CHUNK SAMPLES ---")
for i, pt in enumerate(sample_points, 1):
    payload = pt.get("payload", {})
    text = payload.get("text") or payload.get("page_content") or ""
    repo = payload.get("repo") or payload.get("metadata", {}).get("repo", "unknown")
    source = payload.get("source_file") or payload.get("metadata", {}).get("source_file", "unknown")
    print(f"\n[SAMPLE {i}] Point ID: {pt['id']} | Repo: {repo} | Source: {source}")
    print(f"Text (first 200 chars):\n{text[:200]}")
    print("-" * 50)
