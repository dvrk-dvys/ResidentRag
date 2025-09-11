#!/usr/bin/env bash
set -euo pipefail

echo "🔎 Waiting for Elasticsearch at ${ES_URL:-http://elasticsearch:9200} ..."
until curl -fsS "${ES_URL:-http://elasticsearch:9200}/_cluster/health?wait_for_status=yellow&timeout=30s" >/dev/null; do
  echo "…ES not ready yet"; sleep 2;
done
echo "✅ Elasticsearch reachable"

echo "🔎 Waiting for Qdrant at ${QDRANT_URL:-http://qdrant:6333} ..."
until curl -fsS "${QDRANT_URL:-http://qdrant:6333}/collections" >/dev/null; do
  echo "…Qdrant not ready yet"; sleep 2;
done
echo "✅ Qdrant reachable"

if [ "${SKIP_WARMUP:-false}" = "true" ]; then
  echo "⏭️  SKIP_WARMUP=true — skipping warmup"
else
  echo "🔥 Warmup starting (model + tiny queries)…"
  python -m app.warmup || echo "Warmup failed (non-fatal)"
  echo "✅ Warmup done"
fi

echo "🚀 Starting Streamlit"
exec streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0
