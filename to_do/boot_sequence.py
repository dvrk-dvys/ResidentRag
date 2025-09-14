gotcha — here’s the lean version, no curls, just what to click/open, and the right order.

# 0) Prep data (can be done offline)

```bash
python scripts/load_dataset.py   # or your create_medical_seed.py
```

# 1) Start infra needed for ingest (ES + Qdrant + Postgres)

```bash
docker compose config --quiet
docker compose up -d elasticsearch qdrant postgres
docker compose ps
# optional: logs if you want
# docker logs -f es_medi_rag
# docker logs -f qdrant_medi_rag
```

Open in browser to confirm:

* Elasticsearch: [http://localhost:9200](http://localhost:9200)
* Qdrant: [http://localhost:6333](http://localhost:6333)
* (Postgres has no browser UI; you can skip)

# 2) Run ingesters (from host)

```bash
# ES
ES_URL=http://localhost:9200 ES_INDEX=medical_docs \
python scripts/load_to_elasticsearch.py

# Qdrant
QDRANT_URL=http://localhost:6333 QDRANT_COLLECTION=medical_rag_sparse \
python app/ingestion/qdrant_ingest.py
```

# 3) Start the app layer (Streamlit + Grafana)

```bash
docker compose up -d streamlit grafana
docker compose ps
# optional logs:
# docker logs -f residentrag-streamlit-1
```

Open in browser:

* Streamlit app: [http://localhost:8501](http://localhost:8501)
* Grafana: [http://localhost:3010](http://localhost:3010)

### FAQ

* **Should compose up be last?**
  Do data prep first. Then you **must** have Elasticsearch/Qdrant running before the ingesters. After ingest finishes, bring up Streamlit/Grafana. So the **app** is last, but **ES/Qdrant** must be up before ingestion.

* **Can ingestors run without Docker up?**
  They don’t need Streamlit/Grafana, but they **do** need ES and Qdrant running (your scripts hit `http://localhost:9200` and `http://localhost:6333`). So start **those** containers first.
