#!/usr/bin/env python3
import json
import urllib.request
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_eval"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "

def search(query, limit=5):
    vec = embedder.encode(BGE_PREFIX + query, normalize_embeddings=True).tolist()
    body = {"vector": vec, "limit": limit, "with_payload": True}
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())["result"]

# 1. Entry 04 (FastAPI async route threadpool):
# Let's find the canonical chunk in docs/en/docs/async.md
res_e04 = search("declare path operation function normal def instead of async def threadpool worker blocking", limit=3)
print("E04 best match:", res_e04[0]["id"], res_e04[0]["payload"]["source_file"])

# 2. Entry 07 (N+1 loop query optimization):
# In docs/topics/db/optimization.txt or docs/ref/models/querysets.txt
res_e07 = search("Retrieve related objects efficiently select_related prefetch_related multiple times to retrieve", limit=3)
print("E07 best match:", res_e07[0]["id"], res_e07[0]["payload"]["source_file"])

# 3. Entry 21 (Upgrade Django guide):
res_e21 = search("Read the release notes for each final release familiarize yourself with changes upgrade process", limit=3)
print("E21 best match:", res_e21[0]["id"], res_e21[0]["payload"]["source_file"])

# 4. Entry 24 (Password validation settings):
res_e24 = search("Password validation is configured in the AUTH_PASSWORD_VALIDATORS setting", limit=3)
print("E24 best match:", res_e24[0]["id"], res_e24[0]["payload"]["source_file"])

# 5. Entry 25 (GIS tests command):
res_e25 = search("settings file with spatial backends runtests.py gis_tests", limit=3)
print("E25 best match:", res_e25[0]["id"], res_e25[0]["payload"]["source_file"])
