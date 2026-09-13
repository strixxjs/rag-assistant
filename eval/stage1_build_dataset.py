#!/usr/bin/env python3
"""
eval/stage1_build_dataset.py

Stage 1 script: ingest sample HR docs, fetch chunks from Qdrant,
generate 30 eval questions via Groq (free tier), write dataset.json.
"""
import os
import sys
import json
import uuid
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"
COLLECTION = "allright_brain"
DATASET_PATH = "eval/dataset.json"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# 1. Sample HR knowledge base (Ukrainian)
# ---------------------------------------------------------------------------
DOCS = [
    {
        "title": "Політика відпусток (2024)",
        "source": "https://hr.allright.com/vacation-policy",
        "updated_at": "2024-01-15",
        "text": (
            "Всі штатні співробітники компанії All Right мають право на щорічну оплачувану відпустку тривалістю 24 календарних дні. "
            "Відпустка нараховується з першого дня роботи і стає доступною після 6 місяців безперервного стажу. "
            "Для оформлення відпустки необхідно подати заявку в системі Jira HR не менш ніж за 14 календарних днів до її початку. "
            "Заявку має погодити безпосередній керівник протягом 3 робочих днів. "
            "Невикористані дні відпустки можна переносити на наступний рік у кількості не більше 5 днів. "
            "Додаткова відпустка тривалістю 7 днів надається співробітникам з особливими умовами праці (робота зі шкідливими речовинами, нічні зміни тощо). "
            "Відпустка у зв'язку з одруженням — 3 календарні дні. "
            "Відпустка у зв'язку зі смертю близького родича — 3 календарні дні."
        ),
    },
    {
        "title": "Процедура оформлення лікарняного (2024)",
        "source": "https://hr.allright.com/sick-leave",
        "updated_at": "2024-02-01",
        "text": (
            "Лікарняний лист видається лікарем і повинен бути переданий до відділу кадрів протягом 5 робочих днів після одужання. "
            "Перші 5 днів хвороби оплачує роботодавець у розмірі 100% середньої заробітної плати. "
            "Починаючи з 6-го дня оплату здійснює Фонд соціального страхування за встановленими тарифами. "
            "Якщо хвороба тривала більше 30 днів поспіль, співробітник зобов'язаний надати висновок лікарської комісії. "
            "Дистанційні співробітники повідомляють керівника про хворобу в день настання непрацездатності через Slack або email. "
            "Для відкриття електронного лікарняного необхідно мати підтверджений аккаунт на порталі ДІЯ. "
            "Лікарняний під час відпустки не перериває відпустку, але незакриті дні додаються після одужання."
        ),
    },
    {
        "title": "Онбординг нових співробітників (2024)",
        "source": "https://hr.allright.com/onboarding",
        "updated_at": "2024-03-10",
        "text": (
            "Процес онбордингу в All Right розрахований на перші 90 днів роботи і поділяється на три фази. "
            "Фаза 1 (дні 1–30): знайомство з командою, налаштування облікових записів, проходження обов'язкових навчальних курсів з безпеки інформації та антикорупційної політики. "
            "Фаза 2 (дні 31–60): погодження особистих цілей з керівником, перше 1:1 зустрічі раз на 2 тижні, доступ до продуктових репозиторіїв. "
            "Фаза 3 (дні 61–90): самостійне виконання перших задач, зворотній зв'язок від команди, підтвердження проходження випробувального терміну. "
            "Кожному новачку призначається Buddy — досвідчений колега, який допомагає з першими кроками. "
            "На 30-й і 90-й день проводиться зустріч з HR для оцінки адаптації."
        ),
    },
    {
        "title": "Компенсація та переваги (Бенефіти 2024)",
        "source": "https://hr.allright.com/benefits",
        "updated_at": "2024-01-20",
        "text": (
            "All Right пропонує конкурентний пакет компенсацій та пільг для всіх штатних співробітників. "
            "Медичне страхування: корпоративна програма ДМС з покриттям 200 000 грн на рік, включає стоматологію та офтальмологію. "
            "Спортивний бюджет: 5 000 грн на квартал на спортивні послуги, абонемент до спортзалу або онлайн-тренування. "
            "Навчання: бюджет 20 000 грн на рік на курси, конференції, книги та сертифікації. "
            "Корпоративний ноутбук надається з першого дня роботи і повертається при звільненні. "
            "Обід: компенсація 150 грн на день для офісних співробітників через корпоративну картку Monobank. "
            "Дистанційний формат: для позицій з remote-статусом компанія компенсує інтернет до 500 грн на місяць."
        ),
    },
    {
        "title": "Кодекс поведінки та етика (2024)",
        "source": "https://hr.allright.com/code-of-conduct",
        "updated_at": "2024-02-28",
        "text": (
            "Всі співробітники All Right зобов'язані дотримуватись Кодексу поведінки, який є частиною трудового договору. "
            "Заборонено дискримінацію за ознаками статі, раси, релігії, сексуальної орієнтації, віку чи стану здоров'я. "
            "Конфліктні ситуації вирішуються через медіацію з участю HR Business Partner. "
            "Корупційні дії (хабарництво, відкати) є підставою для негайного розірвання контракту. "
            "Конфіденційна інформація компанії не може передаватись третім особам без письмового дозволу. "
            "Порушення Кодексу фіксуються в системі HR і можуть призвести до дисциплінарного стягнення або звільнення. "
            "Анонімні скарги приймаються через гарячу лінію compliance@allright.com."
        ),
    },
    {
        "title": "Дистанційна робота та гнучкий графік (2024)",
        "source": "https://hr.allright.com/remote-work",
        "updated_at": "2024-03-01",
        "text": (
            "All Right підтримує гібридний формат роботи для більшості посад. "
            "Дистанційний режим доступний після 3 місяців роботи в компанії та за погодженням з керівником. "
            "Мінімальна присутність в офісі для гібридних позицій — 2 дні на тиждень. "
            "Гнучкий початок робочого дня: співробітники можуть обирати час початку роботи між 8:00 і 11:00. "
            "Обов'язковий overlap (перетин онлайн) для всієї команди — з 11:00 до 16:00 за київським часом. "
            "Для повністю дистанційних позицій щоквартальний виїзд до офісу є обов'язковим (оплата проїзду та готелю з боку компанії). "
            "Тимчасовий переїзд за кордон до 90 днів можливий при повідомленні HR за 30 днів."
        ),
    },
    {
        "title": "Процес performance review (OKR, 2024)",
        "source": "https://hr.allright.com/performance-review",
        "updated_at": "2024-01-10",
        "text": (
            "Оцінка ефективності в All Right проводиться двічі на рік — у червні та грудні. "
            "Методологія базується на OKR (Objectives and Key Results), де кожен співробітник встановлює 3–5 ключових результатів на квартал. "
            "Рівень виконання OKR вимірюється від 0.0 до 1.0, де 0.7 вважається успішним результатом. "
            "Performance review включає самооцінку, оцінку від керівника та peer-review від 2–3 колег. "
            "За підсумками review можлива зміна грейду та перегляд компенсації. "
            "Підвищення заробітної плати здійснюється не частіше ніж раз на рік і залежить від результатів review. "
            "Співробітники з систематично низькими результатами (нижче 0.4 два квартали поспіль) переходять в режим Performance Improvement Plan (PIP)."
        ),
    },
    {
        "title": "IT-безпека та захист даних (2024)",
        "source": "https://hr.allright.com/it-security",
        "updated_at": "2024-02-15",
        "text": (
            "Всі пристрої, що підключаються до корпоративної мережі, мають бути зареєстровані в системі MDM (Mobile Device Management). "
            "Використання особистих пристроїв для роботи з конфіденційними даними заборонено без активації MobileIron. "
            "Паролі мають бути не коротшими за 12 символів, містити великі та малі літери, цифри та спецсимволи. "
            "Зміна паролів обов'язкова кожні 90 днів. "
            "Двофакторна автентифікація (2FA) обов'язкова для всіх корпоративних систем. "
            "Підозрілі електронні листи необхідно пересилати на phishing@allright.com для аналізу командою безпеки. "
            "Витік даних або підозра на злам повинні бути повідомлені до security@allright.com протягом 1 години."
        ),
    },
    {
        "title": "Процедура звільнення та offboarding (2024)",
        "source": "https://hr.allright.com/offboarding",
        "updated_at": "2024-03-05",
        "text": (
            "Процес звільнення ініціюється написанням заяви або отриманням повідомлення від роботодавця. "
            "Стандартний строк відпрацювання становить 14 календарних днів, якщо інше не передбачено контрактом. "
            "Протягом останнього тижня співробітник зобов'язаний передати всі справи, документацію та доступи призначеному наступнику. "
            "Корпоративний ноутбук, пропуск та інші матеріальні цінності повертаються в останній робочий день. "
            "Розрахункові виплати (зарплата, компенсація за невикористану відпустку) здійснюються в день звільнення. "
            "Exit interview з HR проводиться за 2–3 дні до останнього дня роботи. "
            "Доступ до всіх корпоративних систем блокується о 18:00 останнього робочого дня."
        ),
    },
    {
        "title": "Корпоративне навчання та розвиток (2024)",
        "source": "https://hr.allright.com/learning-development",
        "updated_at": "2024-02-10",
        "text": (
            "All Right інвестує в розвиток співробітників через корпоративну платформу навчання LMS (Learning Management System). "
            "Обов'язкові курси: антикорупційна політика, безпека інформації, нова версія Privacy Policy — завершуються протягом першого місяця роботи. "
            "Факультативні курси: лідерство, управління проектами, технічні навички — доступні для всіх за погодженням з керівником. "
            "Менторинг: будь-який співробітник може запросити ментора через HR-портал. "
            "Внутрішні Tech Talks проводяться щотижня по четвергах о 17:00. "
            "Зовнішні конференції та курси компенсуються в межах навчального бюджету (20 000 грн/рік). "
            "Після проходження зовнішнього навчання співробітник зобов'язаний провести внутрішній sharing-сесію протягом 2 тижнів."
        ),
    },
]

# ---------------------------------------------------------------------------
# 2. Ingest docs into Qdrant (using OpenAI-compatible embeddings or random vectors)
# ---------------------------------------------------------------------------
def ingest_to_qdrant():
    """Ingest sample docs into Qdrant using random vectors (no OpenAI needed)."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, CollectionInfo
    )
    import numpy as np

    client = QdrantClient(url=QDRANT_URL, prefer_grpc=False,
                          check_compatibility=False, timeout=30)

    # Check if collection exists
    try:
        info = client.get_collection(COLLECTION)
        existing_count = info.points_count
        log.info(f"Collection '{COLLECTION}' exists with {existing_count} points.")
        if existing_count and existing_count > 0:
            log.info("Collection already has data, skipping ingestion.")
            return client
    except Exception:
        log.info(f"Collection '{COLLECTION}' not found, creating...")

    # Create collection with 1536-dim vectors (matches text-embedding-3-small)
    # We use random vectors since we don't have OpenAI key for eval purposes.
    # The chunk IDs and payloads are what matter for the dataset.
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    log.info(f"Collection '{COLLECTION}' created.")

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    points = []
    for doc in DOCS:
        chunks = splitter.split_text(doc["text"])
        log.info(f"Doc '{doc['title']}': {len(chunks)} chunks")
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            # Random 1536-dim vector (unit-normalized)
            vec = np.random.randn(1536).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            points.append(PointStruct(
                id=chunk_id,
                vector=vec.tolist(),
                payload={
                    "text": chunk,
                    "source": doc["source"],
                    "title": doc["title"],
                    "updated_at": doc["updated_at"],
                }
            ))

    client.upsert(collection_name=COLLECTION, points=points)
    log.info(f"Ingested {len(points)} chunks into Qdrant.")
    return client


# ---------------------------------------------------------------------------
# 3. Fetch chunks from Qdrant
# ---------------------------------------------------------------------------
def fetch_chunks(client, limit=200):
    """Scroll through all chunks and return them as list of dicts."""
    from qdrant_client.models import ScrollRequest

    all_chunks = []
    offset = None

    while True:
        result, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=50,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_chunks.extend(result)
        if next_offset is None or len(all_chunks) >= limit:
            break
        offset = next_offset

    log.info(f"Fetched {len(all_chunks)} chunks total.")
    return all_chunks


# ---------------------------------------------------------------------------
# 4. Generate questions via Groq
# ---------------------------------------------------------------------------
def call_groq(prompt: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 300) -> str:
    """Call Groq API directly via httpx (no SDK needed)."""
    import httpx

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.4,
    }
    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"Groq call failed: {e}. Using fallback.")
        return None


def gen_single_question(chunk) -> dict:
    """Generate a single-chunk question from a chunk."""
    payload = chunk.payload
    text = payload.get("text", "")
    title = payload.get("title", "Unknown")

    prompt = f"""Ти — QA-інженер, який складає датасет для оцінки RAG-системи.
На основі наступного фрагменту документу сформулюй одне конкретне технічне питання,
відповідь на яке ПОВНІСТЮ міститься в даному фрагменті.
Також надай коротку відповідь (1-3 речення) українською мовою.

Документ: {title}
Текст фрагменту:
\"\"\"
{text[:600]}
\"\"\"

Відповідай СТРОГО у форматі:
ПИТАННЯ: <питання>
ВІДПОВІДЬ: <відповідь>"""

    result = call_groq(prompt)
    if result and "ПИТАННЯ:" in result and "ВІДПОВІДЬ:" in result:
        try:
            q_part = result.split("ПИТАННЯ:")[1].split("ВІДПОВІДЬ:")[0].strip()
            a_part = result.split("ВІДПОВІДЬ:")[1].strip()
            return {"question": q_part, "answer": a_part}
        except Exception:
            pass

    # Fallback: template-based
    return {
        "question": f"Що описується у документі «{title}» щодо: {text[:80].strip()}...?",
        "answer": text[:300].strip(),
    }


def gen_multi_question(chunk1, chunk2) -> dict:
    """Generate a multi-chunk synthesis question."""
    t1 = chunk1.payload.get("text", "")[:400]
    t2 = chunk2.payload.get("text", "")[:400]
    title1 = chunk1.payload.get("title", "")
    title2 = chunk2.payload.get("title", "")

    prompt = f"""Ти — QA-інженер. Склади одне складне питання, відповідь на яке потребує 
синтезу інформації з обох наведених фрагментів. Відповідь має бути повною та охоплювати обидва аспекти.

Фрагмент 1 ({title1}):
\"\"\"{t1}\"\"\"

Фрагмент 2 ({title2}):
\"\"\"{t2}\"\"\"

Формат відповіді:
ПИТАННЯ: <питання, що потребує обох джерел>
ВІДПОВІДЬ: <повна відповідь>"""

    result = call_groq(prompt, max_tokens=400)
    if result and "ПИТАННЯ:" in result and "ВІДПОВІДЬ:" in result:
        try:
            q_part = result.split("ПИТАННЯ:")[1].split("ВІДПОВІДЬ:")[0].strip()
            a_part = result.split("ВІДПОВІДЬ:")[1].strip()
            return {"question": q_part, "answer": a_part}
        except Exception:
            pass

    return {
        "question": f"Яка різниця між '{title1}' та '{title2}' в компанії All Right?",
        "answer": f"{t1[:150]}... {t2[:150]}",
    }


def gen_edge_question(chunk) -> dict:
    """Generate an advanced edge-case question."""
    payload = chunk.payload
    text = payload.get("text", "")
    title = payload.get("title", "Unknown")

    prompt = f"""Ти — досвідчений QA-інженер. Склади одне СКЛАДНЕ крайове питання до фрагменту.
Питання має тестувати граничні умови: конкретні числа, строки, винятки, заборони, процедури.
Питання НЕ має бути очевидним.

Документ: {title}
Текст:
\"\"\"{text[:600]}\"\"\"

Формат:
ПИТАННЯ: <складне крайове питання>
ВІДПОВІДЬ: <точна відповідь на основі тексту>"""

    result = call_groq(prompt)
    if result and "ПИТАННЯ:" in result and "ВІДПОВІДЬ:" in result:
        try:
            q_part = result.split("ПИТАННЯ:")[1].split("ВІДПОВІДЬ:")[0].strip()
            a_part = result.split("ВІДПОВІДЬ:")[1].strip()
            return {"question": q_part, "answer": a_part}
        except Exception:
            pass

    return {
        "question": f"[Крайовий випадок] Які виняткові умови описані в «{title}»?",
        "answer": text[-300:].strip(),
    }


# ---------------------------------------------------------------------------
# 5. Build dataset.json
# ---------------------------------------------------------------------------
def build_dataset(client, chunks):
    """Build 30-entry evaluation dataset and save to DATASET_PATH."""
    import random
    random.seed(42)

    if len(chunks) < 30:
        log.warning(f"Only {len(chunks)} chunks available, need at least 30.")

    # Group chunks by source for multi-chunk pairs
    by_source = {}
    for c in chunks:
        src = c.payload.get("source", "")
        by_source.setdefault(src, []).append(c)

    # Select diverse chunks - at least one per source if possible
    sources = list(by_source.keys())
    diverse_chunks = []
    for src in sources:
        cs = by_source[src]
        random.shuffle(cs)
        diverse_chunks.extend(cs[:3])  # max 3 per source

    # Pad or trim
    random.shuffle(diverse_chunks)
    all_chunks = chunks[:]
    random.shuffle(all_chunks)

    entries = []

    # --- 15 single-chunk entries (slots 0-14) ---
    log.info("Generating 15 single-chunk questions...")
    for i in range(15):
        chunk = diverse_chunks[i % len(diverse_chunks)]
        qa = gen_single_question(chunk)
        entries.append({
            "id": str(uuid.uuid4()),
            "question": qa["question"],
            "ground_truth_chunk_id": str(chunk.id),
            "ground_truth_answer": qa["answer"],
        })
        log.info(f"  [{i+1}/15] chunk_id={chunk.id}, title={chunk.payload.get('title','?')[:40]}")
        time.sleep(0.5)  # respect Groq rate limits

    # --- 5 multi-chunk entries (slots 15-19) ---
    log.info("Generating 5 multi-chunk questions...")
    multi_pairs = []
    for src, cs in by_source.items():
        if len(cs) >= 2:
            multi_pairs.append((cs[0], cs[1]))
    # Fallback: cross-source pairs
    while len(multi_pairs) < 5:
        i1 = random.randint(0, len(all_chunks) - 1)
        i2 = random.randint(0, len(all_chunks) - 1)
        if i1 != i2:
            multi_pairs.append((all_chunks[i1], all_chunks[i2]))

    for i, (c1, c2) in enumerate(multi_pairs[:5]):
        qa = gen_multi_question(c1, c2)
        entries.append({
            "id": str(uuid.uuid4()),
            "question": qa["question"],
            "ground_truth_chunk_id": [str(c1.id), str(c2.id)],
            "ground_truth_answer": qa["answer"],
        })
        log.info(f"  [{i+1}/5] chunk_ids=[{c1.id}, {c2.id}]")
        time.sleep(0.5)

    # --- 10 edge-case entries (slots 20-29) ---
    log.info("Generating 10 edge-case questions...")
    edge_chunks = all_chunks[15:25] if len(all_chunks) >= 25 else all_chunks[-10:]
    for i, chunk in enumerate(edge_chunks[:10]):
        qa = gen_edge_question(chunk)
        entries.append({
            "id": str(uuid.uuid4()),
            "question": qa["question"],
            "ground_truth_chunk_id": str(chunk.id),
            "ground_truth_answer": qa["answer"],
        })
        log.info(f"  [{i+1}/10] chunk_id={chunk.id}, title={chunk.payload.get('title','?')[:40]}")
        time.sleep(0.5)

    # Write to file
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, indent=2, ensure_ascii=False)

    log.info(f"✅ Dataset saved: {DATASET_PATH} ({len(entries)} entries)")
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not GROQ_API_KEY:
        log.error("GROQ_API_KEY environment variable is not set!")
        sys.exit(1)

    log.info("=== Stage 1: Building RAG Eval Dataset ===")

    # Step 1: Ingest docs
    log.info("--- Step 1: Ingesting sample HR documents ---")
    client = ingest_to_qdrant()

    # Step 2: Fetch chunks
    log.info("--- Step 2: Fetching chunks from Qdrant ---")
    chunks = fetch_chunks(client)

    if not chunks:
        log.error("No chunks found in Qdrant! Ingestion may have failed.")
        sys.exit(1)

    # Step 3: Generate dataset
    log.info("--- Step 3: Generating 30-entry eval dataset via Groq ---")
    entries = build_dataset(client, chunks)

    log.info("=== Done! ===")
    log.info(f"Total entries: {len(entries)}")
    log.info(f"Dataset: {DATASET_PATH}")
