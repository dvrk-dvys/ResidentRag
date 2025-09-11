import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from elasticsearch import Elasticsearch

# from app.evaluation.eval_utils import evaluate
from evaluation.eval_utils import evaluate

ES_URL = os.getenv("ES_URL", "http://elasticsearch:9200")
# ES_URL = os.getenv("ES_URL", "http://localhost:9200")
INDEX = os.getenv("ES_INDEX", "medical_docs")


@dataclass
class Hit:
    #'Hit' as in the hybrid search has found some options
    id: str
    title: str
    text: str
    rrf_score: float = 0.0
    source_type: Optional[str] = None

    # ?ADD  OTHER SCORES HERE TOO?


def wait_for_es(es, timeout: int = 60) -> None:
    import os
    import time

    url = os.getenv("ES_URL", "http://elasticsearch:9200")
    print(f"[DEBUG] Waiting for ES at {url} with timeout {timeout}s...")
    start = time.time()
    attempt = 0
    while time.time() - start < timeout:
        attempt += 1
        elapsed = time.time() - start
        try:
            print(
                f"[DEBUG] Attempt {attempt} ({elapsed:.1f}s): checking cluster health…"
            )
            # returns fast if reachable; raises if not
            es.cluster.health(wait_for_status="yellow", timeout="10s")
            print(
                f"[DEBUG] SUCCESS! ES is yellow+ after {elapsed:.1f}s and {attempt} attempts"
            )
            return
        except Exception as e:
            print(f"[DEBUG] ES not ready: {type(e).__name__}: {e}")
            time.sleep(2)
    raise RuntimeError(f"Elasticsearch not healthy after {timeout}s")


def search_elasticsearch(query, top_k=5, source_type=None):
    """
    Simple BM25 search using Elasticsearch for medical data
    """
    ES_CLIENT = Elasticsearch(ES_URL, request_timeout=30)

    search_query = {
        "size": top_k,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "text"],  # title boost!
                        "type": "best_fields",
                    }
                }
            }
        },
        "_source": ["id", "source_type", "title", "text", "wiki_id", "source", "url"],
    }

    if source_type:
        search_query["query"]["bool"]["filter"] = {"term": {"source_type": source_type}}

    response = ES_CLIENT.search(index=INDEX, body=search_query)

    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        # results.append({
        #    'id': source.get('id', 'unknown'),
        #    'title': source.get('title', ''),
        #    'text': source.get('text', ''),
        #    'source_type': source.get('source_type', ''),
        #    'source': source.get('source', ''),
        #    'score': hit['_score']  # BM25 score
        # })

        results.append(
            Hit(
                id=source.get("id", "unknown"),
                title=source.get("title", ""),
                text=source.get("text", ""),
                source_type=source.get("source_type", ""),
                rrf_score=hit["_score"],
            )
        )

    return results


if __name__ == "__main__":

    ground_truth_path = "//data/evaluation/ground_truth.json"
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)
    gt = [{"query": row["question"], "doc_id": row["doc_id"]} for row in gt_raw]

    top_k = 10
    metrics = evaluate(gt, search_elasticsearch, top_k=top_k)

    hit = metrics[f"Hit@{top_k}"]
    mrr = metrics[f"MRR@{top_k}"]
    map_ = metrics[f"MAP@{top_k}"]
    ndcg = metrics[f"nDCG@{top_k}"]

    print("\n📊 Elastic Search Evaluation Results (aggregate, per-query metrics)")
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

    print("🔍 Testing Medical Elasticsearch Search")
    print("=" * 50)

    for query in test_queries:
        print(f"\n📋 Query: '{query}'")
        try:

            results = search_elasticsearch(query, top_k=3)
            print(f"✅ Found {len(results)} results:")

            if not results:
                print("  (no results)")
            else:
                for i, result in enumerate(results, 1):
                    print(f"\n  {i}. RRF Score: {result.rrf_score:.4f}")
                    print(f"     Source: {result.source_type}")
                    print(f"     Title: {result.title[:80]}...")
                    print(f"     Text: {result.text[:150]}...")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 30)
