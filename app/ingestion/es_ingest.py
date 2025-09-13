# scripts/load_to_elasticsearch.py
import argparse
import json
import os
import time
from typing import Dict, Iterable, List

import numpy as np
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def wait_for_es(es, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if es.ping():
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Elasticsearch did not become ready in time")


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
    for path, source_type in SOURCES:
        for d in load_json_array(path):
            _id = d.get("id", "")
            doc = {
                "id": _id if not PREFIX_IDS else f"{source_type}:{_id}",
                "source_type": source_type,  # derived from file
                "title": d.get("title", ""),
                "text": d.get("text", ""),
                # "wiki_id": d.get("wiki_id", ""),
                "source": d.get("source", ""),
                "url": d.get("url", None),  # only if present
            }
            doc["text_vector"] = embed_title_plus_text(doc["title"], doc["text"])
            yield doc


def ensure_index(wipe: bool):
    exists = es.indices.exists(index=INDEX)
    if wipe:
        print(f"🧹 Wiping index '{INDEX}' (if exists) and recreating…")
        # delete alias with the same name (if any)
        try:
            if es.indices.exists_alias(name=INDEX):
                print(f"🔗 Found alias '{INDEX}', deleting…")
                es.indices.delete_alias(index="*", name=INDEX)
        except Exception as e:
            print(f"(alias delete skipped: {e})")

        # delete index
        es.indices.delete(index=INDEX, ignore_unavailable=True)

        # wait until it's actually gone
        for _ in range(30):
            if not es.indices.exists(index=INDEX) and not es.indices.exists_alias(
                name=INDEX
            ):
                break
            time.sleep(1)

        # now (re)create
        es.indices.create(
            index=INDEX, body=index_settings, timeout="60s", master_timeout="60s"
        )
        return

    if not exists:
        print(f"📦 Creating index '{INDEX}' (did not exist)…")
        es.indices.create(
            index=INDEX, body=index_settings, timeout="60s", master_timeout="60s"
        )
    else:
        print(f"📦 Using existing index '{INDEX}' (will upsert/overwrite by _id).")


def es_knn(query, k=10, num_candidates=1000):
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
    body = {
        "knn": {
            "field": "text_vector",
            "query_vector": qvec,
            "k": k,
            "num_candidates": num_candidates,
        },
        "_source": ["id", "source_type", "title", "text", "wiki_id", "source", "url"],
    }
    return es.search(index=INDEX, body=body)["hits"]["hits"]


def es_bm25(query, k=10, source_type=None):
    body = {
        "query": {
            "bool": {
                "must": {
                    "multi_match": {"query": query, "fields": ["title^2", "text"]}
                },
                "filter": (
                    [{"term": {"source_type": source_type}}] if source_type else []
                ),
            }
        },
        "_source": ["id", "source_type", "title", "text", "wiki_id", "source", "url"],
        "size": k,
    }
    return es.search(index=INDEX, body=body)["hits"]["hits"]


def main():
    total = 0
    for doc in tqdm(iter_docs(), desc="Indexing to ES"):
        es.index(index=INDEX, id=doc["id"], document=doc, request_timeout=100)
        total += 1

        print(
            f"→ Indexed {doc.get('id','<no-id>')} [{doc.get('source_type','?')}] {doc.get('title','')[:80]}"
        )

    es.indices.refresh(index=INDEX)
    print(f"✅ Ingested/updated {total} docs into ES index '{INDEX}'")
    print("Count:", es.count(index=INDEX))


if __name__ == "__main__":
    ES_URL = os.getenv("ES_URL", "http://localhost:9200")
    INDEX = os.getenv("ES_INDEX", "medical_docs")
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims; cosine-friendly
    model = SentenceTransformer(MODEL_NAME)

    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "source_type": {"type": "keyword"},  # wikipedia | textbook | pubmed
                "title": {"type": "text", "analyzer": "english"},
                "text": {"type": "text", "analyzer": "english"},
                # "wiki_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "url": {"type": "keyword"},
                "text_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }

    # unify heterogeneous data under a common shape;can filter by source_type later.
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

    PREFIX_IDS = False  # this id didnt work "textbook:Anatomy_Gray_2"

    es = Elasticsearch(
        ES_URL,
        request_timeout=100,
        retry_on_timeout=True,
        max_retries=5,
        http_compress=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wipe",
        type=str2bool,
        default=False,
        help="If true, delete & recreate the index before ingest; else upsert/extend.",
    )
    args = parser.parse_args()

    print(f"🔧 Connecting to ES at {ES_URL} | index='{INDEX}' | wipe={args.wipe}")
    wait_for_es(es)
    ensure_index(wipe=args.wipe)
    main()

    print([d["_source"]["title"] for d in es_bm25("gross anatomy")])
    print([d["_source"]["title"] for d in es_knn("gross anatomy")])
