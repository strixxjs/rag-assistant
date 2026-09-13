#!/usr/bin/env python3
"""
eval/ingest_rag_eval.py — Stage 2: Ingest public documentation into `rag_eval`.

Embedding model : BAAI/bge-small-en-v1.5 (384-dim, 512-token context, local, free)
Chunking        : RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                  — same params as ingestion.py; only the embedding backend changes.

NOTE: All downstream metrics (hit-rate, latency) are tied to bge-small-en-v1.5.
      Switching the embedding model invalidates prior eval results.
"""

import json
import logging
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QDRANT_URL  = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION  = "rag_eval"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_DIM  = 384
BATCH_SIZE  = 256            # sentences per encode() call
UPSERT_EVERY = 1_000         # points per Qdrant upsert

CLONE_DIR = Path("/tmp/docs_repos")

REPOS = [
    {
        "name":       "fastapi",
        "url":        "https://github.com/tiangolo/fastapi.git",
        "docs_paths": ["docs/en"],   # Markdown
    },
    {
        "name":       "django",
        "url":        "https://github.com/django/django.git",
        "docs_paths": ["docs"],      # reStructuredText (.txt)
    },
    {
        "name":       "flask",
        "url":        "https://github.com/pallets/flask.git",
        "docs_paths": ["docs"],      # reStructuredText (.rst)
    },
    {
        "name":       "httpx",
        "url":        "https://github.com/encode/httpx.git",
        "docs_paths": ["docs"],      # Markdown
    },
]

SUPPORTED_EXTENSIONS = {".md", ".rst", ".txt"}

# ---------------------------------------------------------------------------
# RST Cleaning
# ---------------------------------------------------------------------------

def clean_rst_markup(text: str) -> str:
    # 1. Strip RST directives (e.g. ".. code-block::", ".. note::", ".. warning::")
    text = re.sub(r'^\s*\.\.\s+[a-zA-Z0-9_-]+::[^\n]*', '', text, flags=re.MULTILINE)
    
    # 2. Strip RST hyperlink targets, anchor labels, and comments
    text = re.sub(r'^\s*\.\.\s+_[a-zA-Z0-9_\.-]+:\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\.\.\s+.*$', '', text, flags=re.MULTILINE)
    
    # 3. Strip RST inline roles (e.g. :func:`~flask.make_response`, :ref:`label`)
    def _clean_role(m):
        content = m.group(1).strip()
        if '<' in content and '>' in content:
            content = content.split('<')[0].strip()
        if content.startswith('~'):
            content = content.split('.')[-1]
        return content

    text = re.sub(r':[a-zA-Z0-9_-]+:`([^`]+)`', _clean_role, text)

    # 4. Strip heading underlines/overlines (===, ---, ~~~, ^^^, """)
    text = re.sub(r'^[=\-~^"#*+]{3,}\s*$', '', text, flags=re.MULTILINE)

    # 5. Strip ASCII table borders (+---+---+, |===|===|)
    text = re.sub(r'^\+[-=+]+\+\s*$', '', text, flags=re.MULTILINE)

    # 6. Replace trailing :: (RST code block marker) with :
    text = re.sub(r'(\w+)::\s*$', r'\1:', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*::\s*$', '', text, flags=re.MULTILINE)

    # 7. Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def clone_repo(name: str, url: str, docs_paths: list[str]) -> Path:
    dest = CLONE_DIR / name
    if dest.exists() and any(dest.iterdir()):
        log.info(f"[{name}] already cloned at {dest}, skipping.")
        return dest

    log.info(f"[{name}] cloning {url} …")
    CLONE_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse", url, str(dest)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "sparse-checkout", "set"] + docs_paths,
        check=True,
        capture_output=True,
    )
    log.info(f"[{name}] clone done.")
    return dest


# ---------------------------------------------------------------------------
# File iteration
# ---------------------------------------------------------------------------

def iter_doc_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def safe_read(path: Path) -> str | None:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = path.read_text(encoding=enc)
            if path.suffix.lower() in {".rst", ".txt"}:
                text = clean_rst_markup(text)
            if len(text.strip()) >= 60:
                return text
            return None
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Deferred imports — sentence-transformers and qdrant hang on module init
    # in WSL2 with older grpcio; must import after env is known-good.
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    # ------------------------------------------------------------------
    # 1. Embedding model
    # ------------------------------------------------------------------
    log.info(f"Loading {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)
    log.info("Model ready.")

    # ------------------------------------------------------------------
    # 2. Qdrant collection
    # ------------------------------------------------------------------
    client = QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=False,
        check_compatibility=False,
        timeout=60,
    )

    try:
        client.delete_collection(COLLECTION)
        log.info(f"Deleted existing '{COLLECTION}'.")
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    log.info(f"Created collection '{COLLECTION}' (dim={VECTOR_DIM}, cosine).")

    # ------------------------------------------------------------------
    # 3. Text splitter — same params as ingestion.py
    # ------------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # ------------------------------------------------------------------
    # 4. Clone repos
    # ------------------------------------------------------------------
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    repo_entries = []
    for repo in REPOS:
        path = clone_repo(repo["name"], repo["url"], repo["docs_paths"])
        repo_entries.append((repo["name"], path, repo["docs_paths"]))

    # ------------------------------------------------------------------
    # 5. Ingest
    # ------------------------------------------------------------------
    total_docs   = 0
    total_chunks = 0
    buffer: list[PointStruct] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        client.upsert(collection_name=COLLECTION, points=buffer)
        buffer.clear()

    for repo_name, repo_path, docs_paths in repo_entries:
        repo_docs = 0
        repo_chunks = 0

        for docs_subpath in docs_paths:
            docs_dir = repo_path / docs_subpath
            if not docs_dir.exists():
                log.warning(f"[{repo_name}] {docs_dir} missing — skipping.")
                continue

            files = list(iter_doc_files(docs_dir))
            log.info(f"[{repo_name}] {len(files)} files in {docs_dir}")

            for file_path in files:
                text = safe_read(file_path)
                if not text:
                    continue

                chunks = splitter.split_text(text)
                if not chunks:
                    continue

                # Encode whole file in one call; bge-small handles 512 tokens
                vecs = model.encode(
                    chunks,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()

                for idx, (chunk, vec) in enumerate(zip(chunks, vecs)):
                    buffer.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vec,
                            payload={
                                "text":        chunk,
                                "source_file": str(file_path.relative_to(repo_path)),
                                "repo":        repo_name,
                                "chunk_index": idx,
                            },
                        )
                    )

                repo_docs   += 1
                repo_chunks += len(chunks)
                total_docs  += 1
                total_chunks += len(chunks)

                if len(buffer) >= UPSERT_EVERY:
                    flush_buffer()
                    log.info(
                        f"  [{repo_name}] flushed — "
                        f"repo_docs={repo_docs} repo_chunks={repo_chunks} "
                        f"total_chunks={total_chunks}"
                    )

        log.info(f"[{repo_name}] finished — {repo_docs} docs, {repo_chunks} chunks.")

    flush_buffer()

    # ------------------------------------------------------------------
    # 6. Verify count from Qdrant (authoritative)
    # ------------------------------------------------------------------
    info = client.get_collection(COLLECTION)
    exact_chunks = info.points_count

    log.info("=" * 60)
    log.info(f"Embedding model    : {EMBED_MODEL}")
    log.info(f"Documents ingested : {total_docs}")
    log.info(f"Chunks in rag_eval : {exact_chunks}")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 7. Write summary for downstream scripts
    # ------------------------------------------------------------------
    summary = {
        "collection":          COLLECTION,
        "embed_model":         EMBED_MODEL,
        "vector_dim":          VECTOR_DIM,
        "chunk_size":          500,
        "chunk_overlap":       100,
        "documents_ingested":  total_docs,
        "chunks_in_qdrant":    exact_chunks,
    }
    out = Path("eval/results/ingest_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary → {out}")

    if exact_chunks < 20_000:
        log.warning(
            f"Only {exact_chunks} chunks — below 20 000 target. "
            "Consider adding more repos."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
