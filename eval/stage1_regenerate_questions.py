#!/usr/bin/env python3
"""
eval/stage1_regenerate_questions.py

Re-generates questions for existing dataset.json using Groq with correct model.
Keeps existing chunk_ids, only updates questions and answers.
"""
import os, sys, json, uuid, time, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATASET_PATH = "eval/dataset.json"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "allright_brain"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Available free Groq models (from API check):
MODEL = "qwen/qwen3.8-27b"


def call_groq(prompt: str, max_tokens: int = 400) -> str:
    import httpx
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.4,
    }
    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=body, timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"Groq call failed: {e}")
        return None


def fetch_chunk_text(client, chunk_id: str) -> dict:
    """Fetch a single chunk payload from Qdrant."""
    try:
        results = client.retrieve(
            collection_name=COLLECTION,
            ids=[chunk_id],
            with_payload=True,
        )
        if results:
            return results[0].payload
    except Exception as e:
        log.warning(f"Failed to fetch chunk {chunk_id}: {e}")
    return {}


def gen_single_question(title: str, text: str, is_edge: bool = False) -> dict:
    if is_edge:
        prompt = f"""Ти — QA-інженер RAG-системи. Склади одне СКЛАДНЕ крайове питання до фрагменту.
Тестуй конкретні числа, строки, виключення, заборони — не загальні факти.

Документ: {title}
Текст: \"\"\"{text[:600]}\"\"\"

Відповідай ТІЛЬКИ у форматі (без зайвого тексту):
ПИТАННЯ: <складне питання>
ВІДПОВІДЬ: <точна відповідь>"""
    else:
        prompt = f"""Ти — QA-інженер RAG-системи. Склади одне конкретне технічне питання,
відповідь на яке ПОВНІСТЮ міститься в даному фрагменті. Питання має бути природним,
як запит реального співробітника до HR-бота.

Документ: {title}
Текст: \"\"\"{text[:600]}\"\"\"

Відповідай ТІЛЬКИ у форматі (без зайвого тексту):
ПИТАННЯ: <питання>
ВІДПОВІДЬ: <відповідь 1-3 речення>"""

    result = call_groq(prompt)
    if result and "ПИТАННЯ:" in result and "ВІДПОВІДЬ:" in result:
        try:
            q = result.split("ПИТАННЯ:")[1].split("ВІДПОВІДЬ:")[0].strip()
            a = result.split("ВІДПОВІДЬ:")[1].strip()
            # Remove potential markdown
            q = q.strip("*`\"")
            a = a.strip("*`\"")
            return {"question": q, "answer": a}
        except Exception as ex:
            log.warning(f"Parse error: {ex}, raw: {result[:200]}")

    return None


def gen_multi_question(t1: str, text1: str, t2: str, text2: str) -> dict:
    prompt = f"""Ти — QA-інженер RAG-системи. Склади одне складне питання до HR-бота,
відповідь на яке потребує СИНТЕЗУ інформації з обох фрагментів.

Фрагмент 1 ({t1}): \"\"\"{text1[:400]}\"\"\"
Фрагмент 2 ({t2}): \"\"\"{text2[:400]}\"\"\"

Відповідай ТІЛЬКИ у форматі:
ПИТАННЯ: <питання, що потребує обох джерел>
ВІДПОВІДЬ: <повна відповідь, що синтезує обидва фрагменти>"""

    result = call_groq(prompt, max_tokens=500)
    if result and "ПИТАННЯ:" in result and "ВІДПОВІДЬ:" in result:
        try:
            q = result.split("ПИТАННЯ:")[1].split("ВІДПОВІДЬ:")[0].strip()
            a = result.split("ВІДПОВІДЬ:")[1].strip()
            return {"question": q, "answer": a}
        except Exception as ex:
            log.warning(f"Parse error: {ex}")

    return None


def main():
    if not GROQ_API_KEY:
        log.error("GROQ_API_KEY not set!")
        sys.exit(1)

    # Test Groq connection first
    log.info(f"Testing Groq API with model: {MODEL}")
    test = call_groq("Відповідай одним словом: столиця України?", max_tokens=10)
    if test:
        log.info(f"Groq OK, response: {test[:50]}")
    else:
        log.error("Groq test failed! Check API key and model name.")
        sys.exit(1)

    # Load existing dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data["entries"]
    log.info(f"Loaded {len(entries)} entries from {DATASET_PATH}")

    # Connect to Qdrant
    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=False,
                          check_compatibility=False, timeout=30)

    # Process entries
    total = len(entries)
    updated = 0

    for i, entry in enumerate(entries):
        chunk_ids = entry["ground_truth_chunk_id"]
        log.info(f"[{i+1}/{total}] chunk_id={chunk_ids}")

        if isinstance(chunk_ids, list):
            # Multi-chunk entry
            p1 = fetch_chunk_text(client, chunk_ids[0])
            p2 = fetch_chunk_text(client, chunk_ids[1])
            if p1 and p2:
                qa = gen_multi_question(
                    p1.get("title", "?"), p1.get("text", ""),
                    p2.get("title", "?"), p2.get("text", ""),
                )
                if qa:
                    entry["question"] = qa["question"]
                    entry["ground_truth_answer"] = qa["answer"]
                    updated += 1
                    log.info(f"  → Q: {qa['question'][:80]}...")
        else:
            # Single-chunk entry (edge-case for last 10)
            is_edge = i >= 20
            p = fetch_chunk_text(client, chunk_ids)
            if p:
                qa = gen_single_question(
                    p.get("title", "?"), p.get("text", ""),
                    is_edge=is_edge,
                )
                if qa:
                    entry["question"] = qa["question"]
                    entry["ground_truth_answer"] = qa["answer"]
                    updated += 1
                    log.info(f"  → Q: {qa['question'][:80]}...")

        time.sleep(0.4)  # rate limit

    # Save updated dataset
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, indent=2, ensure_ascii=False)

    log.info(f"✅ Updated {updated}/{total} entries. Saved to {DATASET_PATH}")


if __name__ == "__main__":
    main()
