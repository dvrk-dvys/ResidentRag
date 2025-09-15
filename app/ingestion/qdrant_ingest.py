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


def point_uuid(s: str) -> str:
    # include source in the string so textbook/pubmed don't collide
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{COLLECTION_NAME}:{s}"))


# 2) your existing flush helper (simple + batched skip)
def flush_batch(client, buf, collection_name):
    if not buf:
        return 0, 0

    # use stable UUIDs for IDs
    ids = [point_uuid(p["id"]) for p in buf]

    existing = {
        str(pt.id)
        for pt in client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=False,
            with_vectors=False,
        )
    }
    # compare as strings; pt.id may already be a UUID object
    todo_payloads = [p for p, pid in zip(buf, ids) if pid not in existing]
    skipped = len(buf) - len(todo_payloads)

    if not todo_payloads:
        return 0, skipped

    texts = [
        ((p["title"] or "") + ("\n\n" + p["text"] if p["text"] else "")).strip()
        for p in todo_payloads
    ]
    vecs = model.encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    )

    points = [
        PointStruct(
            id=point_uuid(p["id"]),  # <-- UUID id
            vector=(v.tolist() if isinstance(v, np.ndarray) else v),
            payload=p,  # keep original string id in payload["id"]
        )
        for p, v in zip(todo_payloads, vecs)
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(points), skipped


def upsert_all(client=QdrantClient, batch_size=512):
    print("🚀 Embedding + upserting to Qdrant (skip existing)…", flush=True)

    total_docs = sum(len(load_json_array(p)) for p, _ in SOURCES)
    buffer, upserted, skipped, processed = [], 0, 0, 0
    pbar = tqdm(total=total_docs, desc="Qdrant upsert", unit="doc", dynamic_ncols=True)

    try:
        for payload in iter_docs():
            buffer.append(payload)
            if len(buffer) >= batch_size:
                u, s = flush_batch(client, buffer, COLLECTION_NAME)
                upserted += u
                skipped += s
                processed += len(buffer)
                pbar.update(len(buffer))
                print(
                    f"   • processed={processed} | upserted={upserted} | skipped={skipped}",
                    flush=True,
                )
                buffer.clear()
    except KeyboardInterrupt:
        print("\n⏹️ Cancelled by user.")
    finally:
        # flush remaining (if any) so you don't lose the last partial batch
        if buffer:
            u, s = flush_batch(client, buffer, COLLECTION_NAME)
            upserted += u
            skipped += s
            processed += len(buffer)
            pbar.update(len(buffer))
        pbar.close()
        print(
            f"✅ Qdrant done. processed={processed}, upserted={upserted}, skipped(existing)={skipped}",
            flush=True,
        )


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
            "data/test_seed/medical_textbook_seed_medium.json",
            "textbook",
        ),
        (
            "data/test_seed/medical_pubmed_seed_medium.json",
            "pubmed",
        ),
        #(
        #    "/Users/jordanharris/Code/ResidentRAG/data/small_seed/medical_textbook_seed_small.json",
        #    "textbook",
        #),
        #(
        #    "/Users/jordanharris/Code/ResidentRAG/data/small_seed/medical_pubmed_seed_small.json",
        #    "pubmed",
        #),
        # (
        #    "/Users/jordanharris/Code/ResidentRAG/data/medium_seed/medical_textbook_seed_medium.json",
        #    "textbook",
        # ),
        # (
        #    "/Users/jordanharris/Code/ResidentRAG/data/medium_seed/medical_pubmed_seed_medium.json",
        #    "pubmed",
        # ),
    ]

    PREFIX_IDS = False

    t_start = time.time()
    print("🔧 Initializing...", flush=True)

    qc = QdrantClient(url=QDRANT_URL)
    recreate_collection(qc)
    try:
        upsert_all(qc)
    except KeyboardInterrupt:
        print("\n⏹️ Cancelled by user.")
        try:
            cnt = qc.count(COLLECTION_NAME, exact=True).count
            print(f"Qdrant count in '{COLLECTION_NAME}': {cnt}")
        except Exception as e:
            print(f"(Could not fetch Qdrant count: {e})")

    print(f"⏱️ Done in {time.time() - t_start:.1f}s", flush=True)
