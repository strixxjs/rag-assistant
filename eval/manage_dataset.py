# eval/manage_dataset.py
import json
import uuid
import argparse
import sys
from typing import List, Union
from pydantic import BaseModel, Field, ValidationError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

DATASET_PATH = "eval/dataset.json"

class EvalEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    ground_truth_chunk_id: Union[str, List[str]]
    ground_truth_answer: str

class EvalDataset(BaseModel):
    entries: List[EvalEntry]

def init_template():
    """Генерує базовий шаблон на 30 записів."""
    entries = []
    # 15 слотів для питань, згенерованих з чанків
    for i in range(15):
        entries.append({
            "id": str(uuid.uuid4()),
            "question": f"[ЗГЕНЕРОВАНО] Питання до чанку {i+1}",
            "ground_truth_chunk_id": "TODO_INSERT_CHUNK_ID",
            "ground_truth_answer": "TODO_INSERT_ANSWER"
        })
    
    # 15 слотів для ручних питань (з них 5 охоплюють 2 чанки)
    for i in range(15):
        chunk_mock = ["TODO_CHUNK_ID_1", "TODO_CHUNK_ID_2"] if i < 5 else "TODO_CHUNK_ID"
        entries.append({
            "id": str(uuid.uuid4()),
            "question": f"[РУЧНЕ] Складне питання {i+1}",
            "ground_truth_chunk_id": chunk_mock,
            "ground_truth_answer": "TODO_INSERT_ANSWER"
        })
    
    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump({"entries": entries}, f, indent=2, ensure_ascii=False)
    
    print(f"Шаблон створено у {DATASET_PATH}. Заповни поля TODO.")

def append_entry(question: str, chunk_ids: str, answer: str):
    """Додає новий запис до існуючого JSON."""
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"entries": []}
    
    # Обробка кількох ID через кому
    parsed_chunks = [c.strip() for c in chunk_ids.split(",")]
    if len(parsed_chunks) == 1:
        parsed_chunks = parsed_chunks[0]

    new_entry = EvalEntry(
        question=question,
        ground_truth_chunk_id=parsed_chunks,
        ground_truth_answer=answer
    )
    
    data["entries"].append(new_entry.model_dump())
    
    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Запис успішно додано. ID: {new_entry.id}")

def validate_dataset(collection_name: str, qdrant_url: str = "http://localhost:6333"):
    """Валідує Pydantic схему та перевіряє наявність векторів у Qdrant."""
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        dataset = EvalDataset(**raw_data)
        print("✅ Pydantic валідація пройдена успішно.")
    except (FileNotFoundError, ValidationError) as e:
        print(f"❌ Помилка валідації датасету:\n{e}")
        sys.exit(1)

    client = QdrantClient(url=qdrant_url)
    missing_chunks = []

    for entry in dataset.entries:
        ids_to_check = entry.ground_truth_chunk_id if isinstance(entry.ground_truth_chunk_id, list) else [entry.ground_truth_chunk_id]
        
        for chunk_id in ids_to_check:
            if "TODO" in chunk_id:
                missing_chunks.append(f"Не заповнено TODO для питання '{entry.id}'")
                continue
            
            try:
                result = client.retrieve(collection_name=collection_name, ids=[chunk_id])
                if not result:
                    missing_chunks.append(f"Чанк {chunk_id} не знайдено у Qdrant (Питання ID: {entry.id})")
            except UnexpectedResponse as e:
                missing_chunks.append(f"Помилка Qdrant для {chunk_id}: {e}")

    if missing_chunks:
        print("❌ Знайдено проблеми з chunk_id:")
        for issue in missing_chunks:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("✅ Всі ground_truth_chunk_id успішно знайдені у Qdrant!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Управління Evaluation Dataset")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Створити шаблон dataset.json на 30 записів")
    
    append_parser = subparsers.add_parser("append", help="Додати запис вручну")
    append_parser.add_argument("-q", "--question", required=True)
    append_parser.add_argument("-c", "--chunks", required=True, help="ID чанку (або кілька через кому)")
    append_parser.add_argument("-a", "--answer", required=True)

    val_parser = subparsers.add_parser("validate", help="Провалідувати JSON та перевірити Qdrant")
    val_parser.add_argument("-c", "--collection", required=True, help="Назва колекції в Qdrant")
    val_parser.add_argument("-u", "--url", default="http://localhost:6333", help="URL Qdrant")

    args = parser.parse_args()

    if args.command == "init":
        init_template()
    elif args.command == "append":
        append_entry(args.question, args.chunks, args.answer)
    elif args.command == "validate":
        validate_dataset(args.collection, args.url)
    else:
        parser.print_help()