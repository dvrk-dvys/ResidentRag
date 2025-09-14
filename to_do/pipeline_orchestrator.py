# This directory should contain reusable, production-ready ingestion modules that your main application uses:

#  src/ingestion/
#  ├── __init__.py
#  ├── data_processor.py      # Clean, validate, transform medical data
#  ├── embedding_generator.py # Generate embeddings (reusable across services)
#  ├── es_client.py          # Elasticsearch connection & operations
#  ├── qdrant_client.py      # Qdrant connection & operations
#  ├── batch_processor.py    # Handle large data batches efficiently
#  └── pipeline_orchestrator.py # Coordinate multi-step ingestion
