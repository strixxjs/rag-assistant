#!/usr/bin/env python3
import json
import urllib.request
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_eval"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "

# First update entry 07 in dataset
with open("eval/dataset_with_meta.json") as f:
    entries = json.load(f)["entries"]

# Fix entry 07
entries[6]["ground_truth_chunk_id"] = "60953332-ef9e-4ad3-adb4-a8c60a5b0169"

clean = [{"id": e["id"], "question": e["question"], "ground_truth_chunk_id": e["ground_truth_chunk_id"], "ground_truth_answer": e["ground_truth_answer"]} for e in entries]
with open("eval/dataset.json", "w", encoding="utf-8") as f:
    json.dump({"entries": clean}, f, indent=2, ensure_ascii=False)
with open("eval/dataset_with_meta.json", "w", encoding="utf-8") as f:
    json.dump({"entries": entries}, f, indent=2, ensure_ascii=False)

print("=== AUDITING MISSES AT @20 ===")
misses_at_20 = []

for i, e in enumerate(entries, 1):
    q = BGE_PREFIX + e["question"]
    truth = e["ground_truth_chunk_id"]
    if isinstance(truth, str):
        truth_set = {truth}
    else:
        truth_set = set(truth)
    
    vec = embedder.encode(q, normalize_embeddings=True).tolist()
    body = {"vector": vec, "limit": 20, "with_payload": True}
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())["result"]
    retrieved_ids = [str(r["id"]) for r in res]
    
    hit20 = bool(truth_set & set(retrieved_ids))
    rank = None
    for r_idx, rid in enumerate(retrieved_ids, 1):
        if rid in truth_set:
            rank = r_idx
            break
            
    if not hit20:
        misses_at_20.append((i, e, res))
        print(f"\n[ENTRY {i:02d}] MISS AT @20")
        print(f"  Question: {e['question']}")
        print(f"  Truth ID: {e['ground_truth_chunk_id']}")
        print(f"  Source  : {e.get('_repo')} -> {e.get('_source_file')}")
        # Fetch truth chunk text
        t_req = urllib.request.Request(
            f"{QDRANT_URL}/collections/{COLLECTION}/points",
            data=json.dumps({"ids": list(truth_set), "with_payload": True}).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(t_req) as t_resp:
            t_pts = json.loads(t_resp.read().decode())["result"]
        for tp in t_pts:
            print(f"  Truth chunk text (first 300 chars):")
            print(f"    {tp['payload'].get('text', '')[:300].replace(chr(10), ' ')}")
        print(f"  Top 2 retrieved instead:")
        for r_idx, r in enumerate(res[:2], 1):
            print(f"    #{r_idx} (score {r['score']:.4f}): {r['payload'].get('source_file')} -> {r['payload'].get('text', '')[:150].replace(chr(10), ' ')}")
    else:
        print(f"[ENTRY {i:02d}] HIT (Rank #{rank}) | Q: {e['question'][:60]}...")

print(f"\nTotal misses at @20: {len(misses_at_20)} / {len(entries)}")
