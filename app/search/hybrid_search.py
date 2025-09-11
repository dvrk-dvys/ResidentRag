import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from elasticsearch import Elasticsearch
from evaluation.eval_utils import evaluate
from qdrant_client import QdrantClient
from search.es_search import wait_for_es
from search.qdrant_search import get_client, get_model
from sentence_transformers import SentenceTransformer

# https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-bool-query

# ElasticSearch Config
ES_URL = os.getenv("ES_URL", "http://elasticsearch:9200")
# ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "medical_docs")
ES_CLIENT = Elasticsearch(ES_URL, request_timeout=30)
# wait_for_es(ES_CLIENT, timeout=120)

# Qdrant Config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_rag_sparse")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# QDRANT_CLIENT = QdrantClient(QDRANT_URL, timeout=30, prefer_grpc=False)
# EMBED_MODEL = SentenceTransformer(MODEL_NAME)

#!NOTE: NO NEED FOR ES kNN would be a second semantic signal using the same embedding model. Is redundant w Qdrant

QDRANT_CLIENT = get_client()
EMBED_MODEL = get_model()


@dataclass
class Hit:
    #'Hit' as in the hybrid search has found some options
    id: str
    title: str
    text: str

    # ?ADD  OTHER SCORES HERE TOO?
    # bm25_score: float = 0.0
    # dense_score: float = 0.0
    # reranker_score: float = 0.0
    rrf_score: float = 0.0

    source_type: Optional[str] = None


def ensure_es_ready(timeout: int = 60) -> None:
    wait_for_es(ES_CLIENT, timeout=timeout)


def get_qdrant_ids(query, limit=50):
    # Get top results up to the limit. We usually take more than you plan to display (e.g., 50)
    # so RRF has latitude to promote items that appear high in either list.

    # vec = EMBED_MODEL.encode([query], normalize_embeddings=True)[0].tolist()
    # points = QDRANT_CLIENT.search(collection_name=QDRANT_COLLECTION, query_vector=vec, limit=limit)

    #!!! speed up
    vec = EMBED_MODEL.encode(
        [query], normalize_embeddings=True, batch_size=32, convert_to_numpy=True
    )[0]
    points = QDRANT_CLIENT.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vec.tolist(),
        limit=limit,
        with_payload=["id"],
        with_vectors=False,
        search_params={"hnsw_ef": 96, "exact": False},
    ).points

    doc_ids = []
    for p in points:
        payload_id = p.payload.get("id")  # This should be 'pubmed23n0001_1' etc.
        if payload_id:
            doc_ids.append(payload_id)
    return doc_ids


def medical_query_conditional_boost(
    q: str,
    size: int = 50,
    source_filter: Optional[str] = None,
) -> Dict:
    """
        #On pubmed data we boost the title, on all other data types we boost the text
    PubMed branch: boost title
    Non-PubMed branch (including missing source): boost text
    """
    pubmed_branch = {
        "bool": {
            "filter": {"term": {"source": "pubmed"}},
            "must": {
                "multi_match": {
                    "query": q,
                    "fields": ["title^3", "text"],  #! title boosted here
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            },
        }
    }

    non_pubmed_branch = {
        "bool": {
            "filter": {"bool": {"must_not": {"term": {"source": "pubmed"}}}},
            "must": {
                "multi_match": {
                    "query": q,
                    "fields": ["title", "text^3"],  #! text boosted here
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            },
        }
    }
    bool_query = {
        "should": [pubmed_branch, non_pubmed_branch],
        "minimum_should_match": 1,
    }

    filters: List[Dict] = []
    if source_filter:
        filters.append({"term": {"source": source_filter}})
    if filters:
        bool_query["filter"] = filters

    return {"query": {"bool": bool_query}, "size": size, "_source": ["id"]}


def get_es_ids(
    query: str, size: int = 50, source_type: Optional[str] = None
) -> List[str]:
    # Get top results up to the size (same as limit in qdrant). We usually take more than you plan to display (e.g., 50)
    # so RRF has latitude to promote items that appear high in either list.
    body = medical_query_conditional_boost(query, size=size, source_filter=source_type)
    res = ES_CLIENT.search(index=ES_INDEX, body=body)
    return [(h["_source"].get("id") or h["_id"]) for h in res["hits"]["hits"]]


def es_get_docs(ids):
    # Hydarates all results (both Elastic Search & Qdrant IdS) using elasticsearch,
    # returns a hit object with the actual contents of the search for print out and readability
    if not ids:
        return []
    res = ES_CLIENT.mget(index=ES_INDEX, body={"ids": ids})["docs"]
    out = []
    for d in res:
        if not d.get("found"):
            continue
        s = d.get("_source", {}) or {}
        title = s.get("title")
        text = s.get("text")
        if not title or not text:
            continue  # enforce required fields
        out.append(
            Hit(
                id=s.get("id", d.get("_id")),
                title=title,
                text=text,
                source_type=s.get("source") or s.get("source_type"),
                rrf_score=0.0,  #! set RRF score later
            )
        )
    # preserve the input order of ids
    order = {doc_id: i for i, doc_id in enumerate(ids)}
    out.sort(key=lambda h: order.get(h.id, 10**9))
    return out


def weighted_rrf_fuse(lists, weights=None, k=60, top_k=10):
    # lists: [list_of_ids_from_qdrant, list_of_ids_from_es, ...]
    # weights: e.g., [2.0, 1.0]  -> Qdrant counts double    #Reciprocal Rerank Fusion
    # It merges multiple ranked lists by giving each item a score

    # fuse multiple ranked ID lists -> top_k fused IDs with scores (for later hydration/printing)
    if weights is None:
        weights = [1.0] * len(lists)
    scores = {}
    for L, w in zip(lists, weights):
        for rank, doc_id in enumerate(L, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (k + rank))
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return fused  # list of (doc_id, rrf_score)


def hybrid_search(query: str, top_k: int = 10) -> list[Hit]:
    ensure_es_ready(timeout=30)
    q_ids = get_qdrant_ids(query, limit=50)
    e_ids = get_es_ids(query, size=50)

    fused = weighted_rrf_fuse([q_ids, e_ids], weights=[2.0, 1.0], k=60, top_k=top_k)
    weighted_hybrid_ids = [doc_id for doc_id, _ in fused]
    score_map = dict(fused)

    docs = es_get_docs(weighted_hybrid_ids)

    for d in docs:
        d.rrf_score = score_map.get(d.id, 0.0)

    # Sort RRF scores in descending order
    # reranked_docs = sorted(d.rrf_score.items(), key=lambda x: x[1], reverse=True)
    fused_order = {doc_id: i for i, doc_id in enumerate(weighted_hybrid_ids)}
    docs.sort(key=lambda d: (-d.rrf_score, fused_order.get(d.id, 10**9)))
    reranked_docs = docs[:top_k]
    return reranked_docs


if __name__ == "__main__":

    ground_truth_path = "//data/evaluation/ground_truth.json"
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)
    gt = [{"query": row["question"], "doc_id": row["doc_id"]} for row in gt_raw]

    top_k = 5
    metrics = evaluate(gt, hybrid_search, top_k=top_k)

    hit = metrics[f"Hit@{top_k}"]
    mrr = metrics[f"MRR@{top_k}"]
    map_ = metrics[f"MAP@{top_k}"]
    ndcg = metrics[f"nDCG@{top_k}"]

    print("\n📊 Hybrid Evaluation Results (aggregate, per-query metrics)")
    print(
        f"Hit@{top_k}: {hit:.3f} | MRR@{top_k}: {mrr:.3f} | MAP@{top_k}: {map_:.3f} | nDCG@{top_k}: {ndcg:.3f}"
    )

    test_queries = [
        "How effective is BA 1 immunostimulant compared to ifosfamide for treating carcinosarcoma in rats?",
        "What are the characteristics and symptoms of gastric carcinoma?",
        "What is cardiac muscle and how does its innervation differ from smooth muscle?",
        "What are the different types of blood vessels and what layers make up their walls?",
        "What are the lymphatic drainage pathways of the breast?",
    ]
    print("\n🧾 Hydrated Hybrid Top-3 (title + text)")
    print("=" * 50)
    for query in test_queries:
        print(f"\n📋 Query: {query}")
        try:

            results = hybrid_search(query, top_k=top_k)  # (doc_id, rrf_score)
            print(f"✅ Found {len(results)} results:")

            if not results:
                print("  (no results)")
            else:
                for i, d in enumerate(results, 1):
                    print(f"\n  {i}. RRF Score: {d.rrf_score:.4f}")
                    print(f"     Source: {d.source_type}")
                    print(f"     Title:  {d.title}")
                    snippet = d.text if len(d.text) < 600 else d.text[:600] + "..."
                    print(f"     Text:   {snippet}")
                    print(
                        "---------------------------------------------------------------"
                    )

        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 30)
