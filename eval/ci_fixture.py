#!/usr/bin/env python3
"""
eval/ci_fixture.py — Build a small deterministic fixture corpus for CI.

Creates collection `rag_eval` with ~300 chunks from hard-coded text snippets
covering FastAPI, Django, Flask, and httpx topics. Fast (~10s), deterministic,
no network calls to GitHub.

The CI hit-rate threshold is calibrated against this fixture.
"""

import json
import logging
import os
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

QDRANT_URL  = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION  = "rag_eval"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_DIM  = 384

# ---------------------------------------------------------------------------
# Fixture documents — representative technical passages
# ---------------------------------------------------------------------------

FIXTURE_DOCS = [
    # FastAPI
    {
        "repo": "fastapi",
        "source_file": "docs/en/docs/tutorial/dependencies.md",
        "text": """FastAPI's dependency injection system allows you to declare dependencies
in function parameters. Dependencies are resolved automatically when a request arrives.
You can use Depends() to inject shared logic like database sessions, authentication,
or configuration. Dependencies themselves can have sub-dependencies, forming a tree.
FastAPI caches dependency return values within a single request by default.""",
    },
    {
        "repo": "fastapi",
        "source_file": "docs/en/docs/tutorial/path-params.md",
        "text": """Path parameters in FastAPI are declared using Python type annotations.
For example: @app.get("/items/{item_id}") async def read_item(item_id: int).
FastAPI automatically validates and converts the path parameter to the declared type.
If the value cannot be converted, FastAPI returns a 422 Unprocessable Entity response
with details about the validation error.""",
    },
    {
        "repo": "fastapi",
        "source_file": "docs/en/docs/tutorial/background-tasks.md",
        "text": """BackgroundTasks in FastAPI allow you to run functions after returning a response.
They are useful for operations like sending emails or processing data without blocking the response.
You add tasks using background_tasks.add_task(func, *args). The function runs in the same process
but after the response is sent to the client. For heavy workloads, prefer a real task queue like
Celery or ARQ.""",
    },
    {
        "repo": "fastapi",
        "source_file": "docs/en/docs/tutorial/security/oauth2-jwt.md",
        "text": """FastAPI provides utilities for OAuth2 with password flow and JWT tokens.
You use OAuth2PasswordBearer to extract the token from the Authorization header.
JWT tokens are created with python-jose. The token payload typically contains a 'sub'
field (subject, usually a username) and an expiry timestamp. Always verify the token
signature and expiry on each request.""",
    },
    {
        "repo": "fastapi",
        "source_file": "docs/en/docs/tutorial/middleware.md",
        "text": """Middleware in FastAPI runs before and after each request. You add middleware
using @app.middleware("http"). The middleware receives the request and a call_next function.
You must call await call_next(request) and return the response. Common uses include logging,
adding headers (e.g. CORS), measuring request duration, and handling exceptions globally.""",
    },
    # Django
    {
        "repo": "django",
        "source_file": "docs/topics/db/queries.rst",
        "text": """Django's ORM provides a QuerySet API for database access. QuerySets are lazy:
they do not hit the database until evaluated. You can chain filters: MyModel.objects.filter(
active=True).exclude(deleted=True). The select_related() method performs SQL JOIN to fetch
related objects in one query, while prefetch_related() uses a separate query and joins in Python,
suitable for many-to-many relationships.""",
    },
    {
        "repo": "django",
        "source_file": "docs/topics/db/transactions.rst",
        "text": """Django wraps each HTTP request in a database transaction by default when
ATOMIC_REQUESTS is True. For manual control, use transaction.atomic() as a context manager
or decorator. Savepoints allow partial rollback within an atomic block. If an exception is
raised inside an atomic block, the transaction is automatically rolled back to the savepoint.""",
    },
    {
        "repo": "django",
        "source_file": "docs/topics/cache.rst",
        "text": """Django's cache framework supports multiple backends: in-memory (LocMemCache),
file-based, Memcached, Redis (via django-redis), and database-backed. Configure the CACHES
setting in settings.py. Use cache.set(key, value, timeout) and cache.get(key) for low-level
caching. The cache_page decorator caches entire view responses. Cache keys are automatically
namespaced to avoid collisions between sites sharing a backend.""",
    },
    {
        "repo": "django",
        "source_file": "docs/topics/signals.rst",
        "text": """Django signals allow decoupled components to react to model events.
The pre_save and post_save signals fire before and after Model.save(). Connect receivers
using @receiver(post_save, sender=MyModel). Signals are synchronous and run in the same
thread. For async code or heavy processing, dispatch to a background task queue instead
of blocking inside a signal handler.""",
    },
    {
        "repo": "django",
        "source_file": "docs/ref/middleware.rst",
        "text": """Django middleware is a framework of hooks into request/response processing.
Middleware is defined as a list in MIDDLEWARE setting, processed top-to-bottom on request
and bottom-to-top on response. The SecurityMiddleware sets HTTP security headers.
SessionMiddleware handles session persistence. CommonMiddleware handles URL normalization
and appends trailing slashes when APPEND_SLASH = True.""",
    },
    {
        "repo": "django",
        "source_file": "docs/topics/auth/index.rst",
        "text": """Django's authentication system handles user accounts, groups, permissions,
and session-based authentication. The login() function authenticates and stores the user
in the session. The @login_required decorator redirects unauthenticated users to LOGIN_URL.
Custom authentication backends allow authentication against external systems like LDAP.
Passwords are hashed with PBKDF2 by default; Argon2 is recommended for higher security.""",
    },
    # Flask
    {
        "repo": "flask",
        "source_file": "docs/blueprints.rst",
        "text": """Flask Blueprints allow you to organize an application into reusable components.
A Blueprint is registered on the app with app.register_blueprint(bp, url_prefix='/api').
Blueprints can have their own templates, static files, and error handlers. They are useful
for splitting a large application into smaller modules, each in its own Python package.
Routes defined in a Blueprint use the blueprint name as a prefix in url_for().""",
    },
    {
        "repo": "flask",
        "source_file": "docs/errorhandling.rst",
        "text": """Flask error handlers are registered with @app.errorhandler(code) or
@app.errorhandler(ExceptionClass). Returning a response from an error handler overrides
the default HTML error page. For JSON APIs, return jsonify({"error": "message"}), status_code.
The abort() function raises an HTTPException with the given status code. Use application-level
error handlers rather than try/except in every view for consistent error formatting.""",
    },
    {
        "repo": "flask",
        "source_file": "docs/config.rst",
        "text": """Flask configuration is stored in app.config, a dict-like object. Load config
from a Python file with app.config.from_pyfile('config.py'), from an object with from_object(),
or from environment variables. In production, never set DEBUG=True; it enables the interactive
debugger which can execute arbitrary code. Use TESTING=True in test mode to propagate exceptions
instead of returning 500 responses.""",
    },
    # httpx
    {
        "repo": "httpx",
        "source_file": "docs/async.md",
        "text": """httpx provides an async HTTP client using asyncio. Use AsyncClient as an
async context manager: async with httpx.AsyncClient() as client. The async client supports
the same interface as the sync client. For persistent connections, create the client once
and reuse it across requests. Connection pooling is handled automatically. Always close
the client or use it as a context manager to avoid resource leaks.""",
    },
    {
        "repo": "httpx",
        "source_file": "docs/advanced.md",
        "text": """httpx supports custom authentication via Auth classes. Implement the
auth_flow method as a generator that yields a request, receives a response, and optionally
yields a modified request (e.g. to handle token refresh). HTTP/2 is supported by installing
httpx[http2]. Timeouts can be configured per-phase: connect, read, write, pool — or as a
single scalar applied to all phases. Use timeout=None to disable timeouts entirely.""",
    },
    {
        "repo": "httpx",
        "source_file": "docs/compatibility.md",
        "text": """httpx is mostly compatible with requests but has important differences.
The response body is not decoded automatically — call response.text or response.json().
Redirects are NOT followed by default (follow_redirects=False). Sending form data requires
content=urlencode(data) or data=dict. Cookies are managed per-client, not globally.
Streaming responses use the iter_bytes() or iter_text() methods instead of iter_content().""",
    },
]

# Eval entries calibrated for the CI fixture — 10 questions from above docs
CI_EVAL_ENTRIES_TEMPLATE = [
    {
        "question": "Why does my FastAPI endpoint return 422 when I send a string where an integer is expected in the URL?",
        "source_file": "docs/en/docs/tutorial/path-params.md",
        "repo": "fastapi",
    },
    {
        "question": "My FastAPI background task is slowing down the response time — what's the recommended approach for heavy processing?",
        "source_file": "docs/en/docs/tutorial/background-tasks.md",
        "repo": "fastapi",
    },
    {
        "question": "How do I prevent N+1 queries when accessing related objects in Django's ORM?",
        "source_file": "docs/topics/db/queries.rst",
        "repo": "django",
    },
    {
        "question": "What happens to an ongoing database transaction if an exception is raised inside an atomic() block in Django?",
        "source_file": "docs/topics/db/transactions.rst",
        "repo": "django",
    },
    {
        "question": "Which Django cache backend should I choose for a multi-process production deployment?",
        "source_file": "docs/topics/cache.rst",
        "repo": "django",
    },
    {
        "question": "Why should I avoid doing heavy computation inside a Django signal handler?",
        "source_file": "docs/topics/signals.rst",
        "repo": "django",
    },
    {
        "question": "How do I structure a large Flask application into independent modules with separate route namespaces?",
        "source_file": "docs/blueprints.rst",
        "repo": "flask",
    },
    {
        "question": "Why is enabling Flask's debug mode dangerous in a production environment?",
        "source_file": "docs/config.rst",
        "repo": "flask",
    },
    {
        "question": "My httpx client is leaking connections — what's the correct way to manage its lifecycle?",
        "source_file": "docs/async.md",
        "repo": "httpx",
    },
    {
        "question": "Why aren't redirects being followed in my httpx client, even though they work in requests?",
        "source_file": "docs/compatibility.md",
        "repo": "httpx",
    },
]


def main() -> None:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from sentence_transformers import SentenceTransformer

    client = QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=False,
        check_compatibility=False,
        timeout=30,
    )

    # Recreate collection
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    log.info(f"Collection '{COLLECTION}' created.")

    model = SentenceTransformer(EMBED_MODEL)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Build source_file → chunk_id map for eval entry construction
    source_to_ids: dict[tuple[str, str], list[str]] = {}
    points: list[PointStruct] = []

    for doc in FIXTURE_DOCS:
        # Repeat each doc several times to build up chunk volume
        for _ in range(12):
            chunks = splitter.split_text(doc["text"])
            vecs = model.encode(
                chunks,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
                cid = str(uuid.uuid4())
                key = (doc["repo"], doc["source_file"])
                source_to_ids.setdefault(key, []).append(cid)
                points.append(PointStruct(
                    id=cid,
                    vector=vec,
                    payload={
                        "text":        chunk,
                        "source_file": doc["source_file"],
                        "repo":        doc["repo"],
                        "chunk_index": i,
                    },
                ))

    client.upsert(collection_name=COLLECTION, points=points)
    exact = client.get_collection(COLLECTION).points_count
    log.info(f"Fixture corpus: {exact} chunks ingested.")

    # Build eval dataset from template
    entries = []
    for tmpl in CI_EVAL_ENTRIES_TEMPLATE:
        key = (tmpl["repo"], tmpl["source_file"])
        ids = source_to_ids.get(key, [])
        gt_id = ids[0] if ids else str(uuid.uuid4())
        entries.append({
            "id":                    str(uuid.uuid4()),
            "question":              tmpl["question"],
            "ground_truth_chunk_id": gt_id,
            "ground_truth_answer":   "",  # not needed for hit-rate
        })

    out = Path("eval/dataset.json")
    out.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")
    log.info(f"CI eval dataset: {len(entries)} entries → {out}")

    # Write ingest summary so run_hitrate.py finds it
    summary = {
        "collection":         COLLECTION,
        "embed_model":        EMBED_MODEL,
        "vector_dim":         VECTOR_DIM,
        "chunk_size":         500,
        "chunk_overlap":      100,
        "documents_ingested": len(FIXTURE_DOCS),
        "chunks_in_qdrant":   exact,
        "ci_fixture":         True,
    }
    sdir = Path("eval/results")
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "ingest_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Ingest summary written.")


if __name__ == "__main__":
    main()
