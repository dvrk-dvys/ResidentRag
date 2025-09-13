# Following your development plan's "Qdrant-First Strategy", your script should:
#  - Load the same medical data (6000 docs you already have)
#  - Generate embeddings using the same model as your course example (sentence-transformers/all-MiniLM-L6-v2)
#  - Store vectors with metadata (source_type, title, text, etc.)
#  - Handle the triple-source tagging (wikipedia/textbook/pubmed)
import hashlib
import json
import os
import time
import uuid
from typing import Dict, Iterable, List

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def load_json_array(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_title_plus_text(title: str, text: str) -> List[float]:
    # Creates document embeddings with the same model you’ll use for queries
    combo = (title or "").strip()
    if text:
        combo = (combo + "\n\n" + text).strip() if combo else text
    vec = model.encode([combo], normalize_embeddings=True)[0]
    return vec.tolist() if isinstance(vec, np.ndarray) else vec


def iter_docs() -> Iterable[Dict]:
    """
    Yield payload dicts from different data sources with required  fields.
    """
    print("📥 Loading sources...", flush=True)
    for path, source_type in SOURCES:
        rows = load_json_array(path)  # load once per file
        print(f"  • {source_type:<9} {len(rows):>5} docs  ({path})", flush=True)
        for d in rows:
            _id = d.get("id", "")
            if not _id:
                # Optional: stable fallback if id is missing
                # import hashlib at top if you use this
                import hashlib

                _id = hashlib.sha1(
                    f"{source_type}:{d.get('title', '')}".encode("utf-8")
                ).hexdigest()[:16]

            yield {
                "id": f"{source_type}:{_id}" if PREFIX_IDS else _id,
                "source_type": source_type,
                "title": (d.get("title") or ""),
                "text": (d.get("text") or ""),
                # "wiki_id": (d.get("wiki_id") or ""),
                "source": (d.get("source") or ""),
                "url": d.get("url", None),
            }


def creat_collection(client: QdrantClient):
    # Create the collection with specified sparse vector parameters
    # Sets up a sparse (BM25-like) index inside Qdrant. You don’t need it because Elasticsearch already provides BM25 for the hybrid.
    # !! Stick to dense vectors in Qdrant for this project.
    client.create_collection(
        collection_name=COLLECTION_NAME,
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )


def recreate_collection(client: QdrantClient):
    # recreate the collection
    print(
        f"🧰 (Re)creating Qdrant collection '{COLLECTION_NAME}' [dense 384, cosine]...",
        flush=True,
    )
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )


def upsert_all(client: QdrantClient, batch_size: int = 512):
    # Send the points to the collection
    print("🚀 Embedding + upserting to Qdrant...", flush=True)

    batch, upserted = [], 0

    for payload in tqdm(iter_docs(), desc="Processing docs"):
        vec = embed_title_plus_text(payload["title"], payload["text"])
        batch.append(PointStruct(id=uuid.uuid4().hex, vector=vec, payload=payload))
        upserted += 1

        if len(batch) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            print(f"   • Upserted so far: {upserted}", flush=True)
            batch.clear()

    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"   • Final upsert flush. Total: {upserted}", flush=True)

    print(f"✅ Upserted {upserted} points.\n", flush=True)


if __name__ == "__main__":
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medical_rag_sparse")

    client = QdrantClient(QDRANT_URL)
    client.get_collections()
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, cosine
    model = SentenceTransformer(MODEL_NAME)

    SOURCES = [
        # ("/Users/jordanharris/Code/wiki_rag/data/small_seed/medical_wiki_seed_small.json",     "wikipedia"),
        (
            "/Users/jordanharris/Code/wiki_rag/data/small_seed/medical_textbook_seed_small.json",
            "textbook",
        ),
        (
            "/Users/jordanharris/Code/wiki_rag/data/small_seed/medical_pubmed_seed_small.json",
            "pubmed",
        ),
    ]

    PREFIX_IDS = False

    t_start = time.time()
    print("🔧 Initializing...", flush=True)

    qc = QdrantClient(url=QDRANT_URL)
    recreate_collection(qc)
    upsert_all(qc)
    print(f"⏱️ Done in {time.time() - t_start:.1f}s", flush=True)
