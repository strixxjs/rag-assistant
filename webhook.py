from fastapi import FastAPI, Request
from dotenv import load_dotenv
from rag import ask
import httpx
import os

from ingestion import ingest_document

load_dotenv()

GOOGLE_TOKEN = os.getenv("GOOGLE_TOKEN", "")

app = FastAPI()


def extract_text(doc: dict) -> str:
    """
    Витягує текст з Google Docs API response.
    Спрощена реалізація — для концепту.
    У продакшені потрібно обробляти всі типи елементів
    (таблиці, зображення, списки, вкладені структури).
    """
    text_parts = []
    for element in doc.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        for run in paragraph.get("elements", []):
            text_run = run.get("textRun")
            if not text_run:
                continue
            content = text_run.get("content", "")
            if not content.strip():
                continue
            text_parts.append(content.rstrip("\n"))
    return "".join(text_parts)


@app.post("/webhook/google-drive")
async def handle_drive_change(request: Request):
    data = await request.json()
    file_id = data.get("fileId")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://docs.googleapis.com/v1/documents/{file_id}",
            headers={"Authorization": f"Bearer {GOOGLE_TOKEN}"}
        )
        doc = response.json()
        text = extract_text(doc)
        source_url = f"https://docs.google.com/document/d/{file_id}"
        title = doc["title"]

    ingest_document(text, source_url, title)
    return {"status": "reindexed", "file_id": file_id}

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "")
    if not question:
        return {"error": "question is required"}
    answer = ask(question)
    return {"answer": answer}