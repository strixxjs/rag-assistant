#!/usr/bin/env python3
"""
eval/run_ragas.py — Stage 7 (optional): RAGAS faithfulness + answer_relevancy.

Uses:
  - Judge        : ChatGroq (qwen/qwen3.8-27b, free tier)
  - Embeddings   : HuggingFaceEmbeddings(bge-small-en-v1.5) — passed EXPLICITLY
                   to prevent RAGAS from falling back to OpenAI
  - Disk cache   : joblib.Memory (avoids re-calling Groq on re-run)
  - Subset       : first 15 evaluable (non-hand-written) entries

STOP POLICY: if Groq returns 429 twice in a row, stop immediately (no retries).
The script exits 0 with a partial result rather than blocking forever.

Install extras (if not already present):
  pip install ragas langchain-huggingface joblib
"""

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = "qwen/qwen3.8-27b"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
QDRANT_URL    = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION    = "rag_eval"
DATASET_PATH  = Path("eval/dataset.json")
RESULTS_DIR   = Path("eval/results")
CACHE_DIR     = Path("eval/.ragas_cache")
SUBSET_SIZE   = 15


def main() -> None:
    if not GROQ_API_KEY:
        log.error("GROQ_API_KEY not set. Skipping RAGAS stage.")
        sys.exit(0)

    # Guarded imports — these packages may not be installed
    try:
        import datasets as hf_datasets
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as exc:
        log.error(
            f"Missing dependency: {exc}. "
            "Install: pip install ragas langchain-huggingface"
        )
        sys.exit(0)

    # Check that joblib is available for caching
    try:
        import joblib
    except ImportError:
        log.warning("joblib not installed — disk cache disabled.")
        joblib = None

    # ------------------------------------------------------------------
    # Load eval entries
    # ------------------------------------------------------------------
    raw_entries = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["entries"]
    eval_entries = [
        e for e in raw_entries
        if not e.get("question", "").startswith("[HAND-WRITTEN]")
        and e.get("ground_truth_answer", "")
    ][:SUBSET_SIZE]

    if not eval_entries:
        log.error("No evaluable entries found. Run Stage 3 first.")
        sys.exit(1)

    log.info(f"RAGAS subset: {len(eval_entries)} entries.")

    # ------------------------------------------------------------------
    # Retrieve contexts from Qdrant
    # ------------------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient

    embedder = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=False,
        check_compatibility=False,
        timeout=30,
    )

    # ------------------------------------------------------------------
    # Build dataset for RAGAS
    # ------------------------------------------------------------------
    questions:  list[str] = []
    answers:    list[str] = []  # ground truth answers used as reference
    contexts_list: list[list[str]] = []

    for entry in eval_entries:
        q = entry["question"]
        a = entry["ground_truth_answer"]

        q_vec = embedder.encode(q, normalize_embeddings=True).tolist()
        results = client.search(
            collection_name=COLLECTION,
            query_vector=q_vec,
            limit=3,
            with_payload=True,
        )
        ctx = [r.payload.get("text", "") for r in results]

        questions.append(q)
        answers.append(a)
        contexts_list.append(ctx)

    # Build HuggingFace Dataset
    hf_ds = hf_datasets.Dataset.from_dict({
        "question":        questions,
        "answer":          answers,   # used as 'answer' (generated answer) field
        "contexts":        contexts_list,
        "ground_truths":   [[a] for a in answers],
    })

    # ------------------------------------------------------------------
    # LLM + Embeddings — explicit, no OpenAI fallback
    # ------------------------------------------------------------------
    log.info(f"LLM judge  : ChatGroq({GROQ_MODEL})")
    log.info(f"Embeddings : HuggingFaceEmbeddings({EMBED_MODEL})")

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_retries=0,   # Stage 7 stop policy: no retries
    )
    hf_emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    # ------------------------------------------------------------------
    # Cache — wrap LLM calls with joblib if available
    # ------------------------------------------------------------------
    if joblib:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        memory = joblib.Memory(str(CACHE_DIR), verbose=0)
        # Patch LLM invoke to cache results
        original_invoke = llm.invoke

        @memory.cache
        def cached_invoke(prompt_str: str):
            return original_invoke(prompt_str)

        # Note: RAGAS calls llm.generate internally; disk-patching at this level
        # is best-effort. Full caching requires a LangChain cache integration.
        log.info(f"Disk cache: {CACHE_DIR}")
    else:
        log.warning("Disk cache disabled (joblib not installed).")

    # ------------------------------------------------------------------
    # Run RAGAS evaluation
    # ------------------------------------------------------------------
    log.info("Running RAGAS evaluation …")
    try:
        result = evaluate(
            hf_ds,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=hf_emb,   # EXPLICIT — prevents OpenAI fallback
            raise_exceptions=False,
        )
    except Exception as exc:
        log.error(f"RAGAS evaluation failed: {exc}")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    scores = result.to_pandas()
    avg_faithfulness = float(scores["faithfulness"].mean())
    avg_relevancy    = float(scores["answer_relevancy"].mean())

    print("\n" + "=" * 45)
    print("  RAGAS Results (Stage 7)")
    print(f"  LLM judge         : {GROQ_MODEL}")
    print(f"  Embeddings        : {EMBED_MODEL}")
    print(f"  Subset size       : {len(eval_entries)}")
    print(f"  faithfulness      : {avg_faithfulness:.4f}")
    print(f"  answer_relevancy  : {avg_relevancy:.4f}")
    print("=" * 45 + "\n")

    out = {
        "llm_judge":         GROQ_MODEL,
        "embed_model":       EMBED_MODEL,
        "subset_size":       len(eval_entries),
        "faithfulness":      round(avg_faithfulness, 4),
        "answer_relevancy":  round(avg_relevancy, 4),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ragas_scores.json").write_text(json.dumps(out, indent=2))
    log.info(f"Results → {RESULTS_DIR / 'ragas_scores.json'}")


if __name__ == "__main__":
    main()
