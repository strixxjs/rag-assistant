import sys
import time

def log(msg):
    with open('/tmp/import_test.log', 'a') as f:
        f.write(f"{time.time():.2f}: {msg}\n")
    print(msg, flush=True)

log("start")

log("importing portalocker...")
import portalocker
log("portalocker OK")

log("importing qdrant_client...")
from qdrant_client import QdrantClient
log("QdrantClient OK")

log("creating client...")
client = QdrantClient(url='http://localhost:6333', prefer_grpc=False, timeout=5)
log("client created")

cols = client.get_collections()
log(f"Collections: {[c.name for c in cols.collections]}")
