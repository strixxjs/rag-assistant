#!/usr/bin/env python3
"""
eval/run_hitrate.py — Stages 4 & 5: Hit-rate evaluation.

Usage:
  # Baseline (dense retrieval only):
  python eval/run_hitrate.py

  # With cross-encoder reranking (Stage 5):
  python eval/run_hitrate.py --rerank

Metrics: hit-rate@1, @3, @5  (exact chunk ID match, no LLM)
Latency: p50 and p95 wall-clock per query (embed + search [+ rerank])

Output:
  eval/results/baseline_hitrate.json   (without --rerank)
  eval/results/reranked_hitrate.json   (with --rerank)
  eval/results/comparison.json         (written when both exist)

Embedding model: read from eval/results/ingest_summary.json
Cross-encoder  : cross-encoder/ms-marco-MiniLM-L-6-v2  (local, free)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

QDRANT_URL   = "http://localhost:6333"
COLLECTION   = "rag_eval"
DATASET_PATH = Path("eval/dataset.json")
RESULTS_DIR  = Path("eval/results")
SUMMARY_PATH = RESULTS_DIR / "ingest_summary.json"

BGE_PREFIX   = "Represent this sentence for searching relevant passages: "
TOP_K_RETRIEVE = 20   # retrieve this many before optional reranking
HITRATE_KS     = [1, 3, 5, 10, 20]
SANITY_THRESHOLD = 0.85   # hit@3 above this → eval set too easy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ingest_summary() -> dict:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found. Run eval/ingest_rag_eval.py first."
        )
    return json.loads(SUMMARY_PATH.read_text())


def load_dataset() -> list[dict]:
    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return raw["entries"]


def ground_truth_ids(entry: dict) -> set[str]:
    gids = entry["ground_truth_chunk_id"]
    if isinstance(gids, list):
        return {str(g) for g in gids}
    return {str(gids)}


def is_hand_written(entry: dict) -> bool:
    return entry.get("question", "").startswith("[HAND-WRITTEN]")


def hit_at_k(retrieved_ids: list[str], truth_ids: set[str], k: int) -> bool:
    return bool(truth_ids & set(retrieved_ids[:k]))


def percentile(values: list[float], p: int) -> float:
    return float(np.percentile(values, p))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG hit-rate evaluation")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank top-20 with cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    summary = load_ingest_summary()
    embed_model_name = summary["embed_model"]
    log.info(f"Embedding model : {embed_model_name}")
    log.info(f"Collection      : {COLLECTION}")
    log.info(f"Reranking       : {args.rerank}")

    embedder = SentenceTransformer(embed_model_name)
    client   = QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=False,
        check_compatibility=False,
        timeout=30,
    )

    cross_encoder = None
    if args.rerank:
        from sentence_transformers import CrossEncoder
        ce_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        log.info(f"Loading cross-encoder: {ce_model}")
        cross_encoder = CrossEncoder(ce_model)
        log.info("Cross-encoder ready.")

    entries = load_dataset()
    eval_entries = [e for e in entries if not is_hand_written(e)]
    log.info(
        f"Dataset: {len(entries)} total, {len(eval_entries)} evaluable "
        f"({len(entries)-len(eval_entries)} hand-written skipped)"
    )

    # ------------------------------------------------------------------
    # Run queries
    # ------------------------------------------------------------------
    latencies: list[float] = []
    hits: dict[int, list[bool]] = {k: [] for k in HITRATE_KS}

    for i, entry in enumerate(eval_entries):
        question = entry["question"]
        truth    = ground_truth_ids(entry)

        t0 = time.perf_counter()

        # 1. Embed question with instruction prefix
        q_vec = embedder.encode(
            BGE_PREFIX + question,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # 2. Search Qdrant
        retrieve_k = TOP_K_RETRIEVE if cross_encoder else max(HITRATE_KS)
        results = client.search(
            collection_name=COLLECTION,
            query_vector=q_vec,
            limit=retrieve_k,
            with_payload=bool(cross_encoder),  # need text for reranking
        )
        retrieved_ids = [str(r.id) for r in results]

        # 3. Rerank (optional)
        if cross_encoder and results:
            pairs = [(question, r.payload.get("text", "")) for r in results]
            scores = cross_encoder.predict(pairs)
            ranked = sorted(
                zip(retrieved_ids, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            retrieved_ids = [rid for rid, _ in ranked]

        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        for k in HITRATE_KS:
            hits[k].append(hit_at_k(retrieved_ids, truth, k))

        if (i + 1) % 5 == 0:
            log.info(
                f"  [{i+1}/{len(eval_entries)}] "
                f"hit@3={sum(hits[3])/(i+1):.2f}  lat={elapsed*1000:.0f}ms"
            )

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    n = len(eval_entries)
    hit_rates = {f"hit_rate@{k}": round(sum(hits[k]) / n, 4) for k in HITRATE_KS}
    lat_p50 = round(percentile(latencies, 50) * 1000, 1)
    lat_p95 = round(percentile(latencies, 95) * 1000, 1)

    result = {
        "mode":           "reranked" if args.rerank else "baseline",
        "embed_model":    embed_model_name,
        "rerank_model":   "cross-encoder/ms-marco-MiniLM-L-6-v2" if args.rerank else None,
        "collection":     COLLECTION,
        "n_eval":         n,
        "n_skipped_hw":   len(entries) - n,
        **hit_rates,
        "latency_p50_ms": lat_p50,
        "latency_p95_ms": lat_p95,
    }

    # ------------------------------------------------------------------
    # Near-duplicate miss analysis (misses at top-3)
    # ------------------------------------------------------------------
    near_duplicate_misses = 0
    miss_details = []
    
    import re
    def calc_overlap(text1: str, text2: str) -> float:
        w1 = set(re.findall(r'\w+', text1.lower()))
        w2 = set(re.findall(r'\w+', text2.lower()))
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / min(len(w1), len(w2))

    for i, entry in enumerate(eval_entries):
        if not hits[3][i]:  # Missed at top-3
            truth = ground_truth_ids(entry)
            t_pts = client.retrieve(collection_name=COLLECTION, ids=list(truth), with_payload=True)
            truth_texts = [p.payload.get("text", "") for p in t_pts if p.payload]
            
            q_vec = embedder.encode(BGE_PREFIX + entry["question"], normalize_embeddings=True).tolist()
            top3_res = client.search(collection_name=COLLECTION, query_vector=q_vec, limit=3, with_payload=True)
            top3_texts = [r.payload.get("text", "") for r in top3_res if r.payload]
            
            is_near_dup = False
            for tt in truth_texts:
                for rt in top3_texts:
                    if calc_overlap(tt, rt) >= 0.70:
                        is_near_dup = True
                        break
                if is_near_dup:
                    break
            
            if is_near_dup:
                near_duplicate_misses += 1
                miss_details.append({
                    "question": entry["question"],
                    "reason": "near-duplicate found in top-3 with >=70% token overlap"
                })

    result["near_duplicate_misses_at_3"] = near_duplicate_misses
    result["total_misses_at_3"] = n - sum(hits[3])

    # ------------------------------------------------------------------
    # Print per-entry breakdown
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  PER-ENTRY DIAGNOSTIC BREAKDOWN")
    print("=" * 90)
    print(f"  {'#':<3} {'@1':<5} {'@3':<5} {'@5':<5} {'@10':<5} {'@20':<5} {'Question'}")
    print("  " + "-" * 86)
    for i, entry in enumerate(eval_entries):
        h1 = "HIT" if hits[1][i] else "miss"
        h3 = "HIT" if hits[3][i] else "miss"
        h5 = "HIT" if hits[5][i] else "miss"
        h10 = "HIT" if hits[10][i] else "miss"
        h20 = "HIT" if hits[20][i] else "miss"
        q_disp = (entry['question'][:55] + "...") if len(entry['question']) > 55 else entry['question']
        print(f"  {i+1:02d} {h1:<5} {h3:<5} {h5:<5} {h10:<5} {h20:<5} {q_disp}")
    print("=" * 90)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print(f"  Mode                  : {result['mode']}")
    print(f"  Embedding model       : {embed_model_name}")
    print(f"  Eval set              : {n} questions")
    for k in HITRATE_KS:
        print(f"  hit-rate@{k}            : {hit_rates[f'hit_rate@{k}']:.4f}")
    print(f"  Near-dup misses (@3)  : {near_duplicate_misses} / {result['total_misses_at_3']} misses")
    print(f"  Latency p50           : {lat_p50} ms")
    print(f"  Latency p95           : {lat_p95} ms")
    print("=" * 55 + "\n")

    # ------------------------------------------------------------------
    # Sanity check (baseline only)
    # ------------------------------------------------------------------
    if not args.rerank and hit_rates["hit_rate@3"] > SANITY_THRESHOLD:
        log.warning(
            f"⚠  hit-rate@3 = {hit_rates['hit_rate@3']:.4f} > {SANITY_THRESHOLD} — "
            "eval set may be too easy. Consider rephrasing questions."
        )
        harder = [
            e["question"] for e in eval_entries
            if hit_at_k(
                [str(r.id) for r in client.search(
                    collection_name=COLLECTION,
                    query_vector=embedder.encode(
                        e["question"], normalize_embeddings=True
                    ).tolist(),
                    limit=3,
                )],
                ground_truth_ids(e),
                3,
            )
        ][:5]
        log.warning("First 5 easily-retrieved questions (consider rephrasing):")
        for q in harder:
            log.warning(f"  • {q[:100]}")

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = "reranked_hitrate.json" if args.rerank else "baseline_hitrate.json"
    out_path = RESULTS_DIR / out_name
    out_path.write_text(json.dumps(result, indent=2))
    log.info(f"Results → {out_path}")

    # Write comparison if both files exist
    baseline_path = RESULTS_DIR / "baseline_hitrate.json"
    reranked_path = RESULTS_DIR / "reranked_hitrate.json"
    if baseline_path.exists() and reranked_path.exists():
        b = json.loads(baseline_path.read_text())
        r = json.loads(reranked_path.read_text())
        comparison = {
            "embed_model":    b["embed_model"],
            "rerank_model":   r["rerank_model"],
            "n_eval":         b["n_eval"],
            "baseline": {
                "hit_rate@1":      b["hit_rate@1"],
                "hit_rate@3":      b["hit_rate@3"],
                "hit_rate@5":      b["hit_rate@5"],
                "latency_p50_ms":  b["latency_p50_ms"],
                "latency_p95_ms":  b["latency_p95_ms"],
            },
            "reranked": {
                "hit_rate@1":      r["hit_rate@1"],
                "hit_rate@3":      r["hit_rate@3"],
                "hit_rate@5":      r["hit_rate@5"],
                "latency_p50_ms":  r["latency_p50_ms"],
                "latency_p95_ms":  r["latency_p95_ms"],
            },
        }
        comp_path = RESULTS_DIR / "comparison.json"
        comp_path.write_text(json.dumps(comparison, indent=2))
        log.info(f"Comparison → {comp_path}")

        print("\n  COMPARISON TABLE")
        print(f"  {'Metric':<20} {'Baseline':>12} {'Reranked':>12}")
        print("  " + "-" * 46)
        for k in HITRATE_KS:
            key = f"hit_rate@{k}"
            print(f"  {key:<20} {b[key]:>12.4f} {r[key]:>12.4f}")
        print(f"  {'latency_p50_ms':<20} {b['latency_p50_ms']:>12.1f} {r['latency_p50_ms']:>12.1f}")
        print(f"  {'latency_p95_ms':<20} {b['latency_p95_ms']:>12.1f} {r['latency_p95_ms']:>12.1f}")
        print()


if __name__ == "__main__":
    main()
