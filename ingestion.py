from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from datetime import datetime
from dotenv import load_dotenv
import uuid
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "allright_brain")


def split_document_text(
    text: str, chunk_size: int = 500, chunk_overlap: int = 100
) -> list[str]:
    if text is None or not str(text).strip():
        raise ValueError("document text must be non-empty")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


def ingest_document(text: str, source_url: str, title: str, updated_at: str = None):
    chunks = split_document_text(text)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents(chunks)

    client = QdrantClient(url=QDRANT_URL)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk,
                "source": source_url,
                "title": title,
                "updated_at": updated_at or datetime.utcnow().isoformat()
            }
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=COLLECTION, points=points)