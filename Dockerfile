FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app
    #HF_DATASETS_CACHE=/hf_cache  #went with streaming instead because the dataset is too big and takes too long

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# App code
COPY app/ ./app/
#COPY *.py ./
COPY app/images/ ./app/images/


# Small seed dataset for offline eval
#COPY .data/small_seed/ .data/small_seed/
#COPY .data/evaluation/ .data/evaluation/
RUN mkdir -p /app/data/small_seed /app/data/evaluation
COPY data/small_seed/medical_textbook_seed_small.json /app/data/small_seed/
COPY data/small_seed/medical_pubmed_seed_small.json   /app/data/small_seed/
COPY data/evaluation/ground_truth.json /app/data/evaluation/

# Entrypoint
#CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]


COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["/app/entrypoint.sh"]
