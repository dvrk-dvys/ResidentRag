# 🏥 ResidentRAG — Medical AI Assistant

![ResidentRAG Medical Assistant](app/images/streamlit/medibot.jpg)

Evidence-based medical answers with citations from **PubMed**, **medical textbooks**, and **Wikipedia** using a hybrid retrieval-augmented generation (RAG) system.

## 📋 Table of Contents
- [Problem Description](#problem-description)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start-docker--recommended)
- [Data Ingestion](#data-download--seed--ingest)
- [Usage](#app-usage)
- [Monitoring](#monitoring-grafana)
- [Technologies](#agentic-tool-technologies)


---

## ❓ Problem Description

Medical knowledge is vast, scattered, and often inaccessible. Clinicians, researchers, and patients need **concise, accurate, cited answers** quickly. ResidentRAG solves this by:

- Combining **semantic search** (Qdrant vectors) with **lexical search** (Elasticsearch BM25)
- Using **Reciprocal Rank Fusion (RRF)** to rerank results
- Iteratively deciding whether to answer directly, search local corpora, or escalate to using agentic Tools to retreive, parse & chunk, then rerank external data sources (Wikipedia and PubMed)

> *"This application implements an agent-based retrieval-augmented generation (RAG) system. The agent iterates up to three times, making decisions at each step about the best strategy to answer the user’s question. On the first pass, it determines whether the query can be answered directly by the base OpenAI LLM. If additional knowledge is needed, it searches the local knowledge base—data pre-upserted into Elasticsearch (lexical search) and Qdrant (vector search). If the local knowledge base is insufficient, and official citations are required, the agent escalates to external tools: the Wikipedia API and PubMed via Bio.Entrez. These external sources are retrieved, chunked, searched, and then reranked together with the local hybrid search results using Reciprocal Rank Fusion (RRF). The system finally returns a concise, medically informed answer with verified citations, ensuring the response is grounded in real scientific literature and documentation."*

## 🏗️ System Architecture

```
User
  |
  v
Streamlit UI
  |
  v
LLM / Agentic Router
  |
  v
Hybrid Tool (FIRST)
  ├─ Elasticsearch
  └─ Qdrant
  |
  v
(Optional) Wikipedia Tool
(Optional) PubMed Tool
  |
  v
LLM Synthesis (grounded answer)
  |
  v
Streamlit (renders answer + citations)
  |
  v
User Feedback (thumbs up/down, notes)
  |
  v
PostgreSQL (feedback table)
  |
  v
Grafana (visualize ratings, CTR, QA health)
```

---

## ⚙️ Prerequisites

- **Docker** and **Docker Compose**
- **OpenAI API key**
- (Optional for localp dev): Python 3.10+ and `pip`

---

## 🔐 Environment

Copy the provided `.env.example` file and fill it out with your own API keys and settings:

```bash
cp .env.example .env
# Then edit .env with your OpenAI API key, email for PubMed access, etc.

#optional: for local testing and running outside f Docker
pip install -r requirements.txt
```

The `.env.example` file contains all necessary environment variables with placeholder values.

---

## 🚀 Quick Start (Docker — recommended)

```bash
# 1) Clone
git clone https://github.com/dvrk-dvys/ResidentRag
cd ResidentRAG

# 2) Env
cp .env.example .env   # if you have one; otherwise create .env from above

# 3) Start stack
docker compose up -d

# 4) Follow logs until Streamlit says "Running on ..."
docker compose logs -f streamlit
```

**Open:**

* **App** → [http://localhost:8501](http://localhost:8501)
* **Grafana** → [http://localhost:3010](http://localhost:3010) (login: `admin/admin` unless changed)
* **Elasticsearch** → [http://localhost:9200](http://localhost:9200)

ℹ️ Elasticsearch can take ~1–2 minutes to turn healthy on first run.

**📸 Screenshot Placeholder:** *Terminal output of `docker compose logs -f streamlit` showing app warmup and tool selection*

## 🎯 Application Interface

The ResidentRAG interface provides an intuitive chat experience with user type selection, response detail preferences, and integrated feedback collection.

![ResidentRAG Interface](/app/images/read_me/medirag_app.png)

**Key Features:**
- **User Type Selection**: Choose between Patient, Healthcare Provider, or Medical Researcher
- **Response Detail Control**: Simple, Detailed, or Technical explanations
- **👍/👎 Feedback Buttons**: Rate response quality for continuous improvement
- **Citation Display**: View sources and references below each response

---

## 📥 Data: Download / Seed / Ingest

**1) Create seed datasets (Hugging Face MedRAG)**

Script: `scripts/load_dataset.py`

```bash
# Example: PubMed medium seed
python scripts/load_dataset.py
# (The script calls create_medical_seed(...) like:)
# create_medical_seed(dataset_path="MedRAG/pubmed",
#   seed_size=60000, output_path="data/medium_seed/test", output_format="json", source="pubmed")
```

![Load Data from HuggingFace](/app/images/read_me/Load_data_from_HF.png)

**2) Ingest to Elasticsearch**

Script: `scripts/load_to_elasticsearch.py`

```bash
# Local host when running outside Docker:
export ES_URL=http://localhost:9200
python scripts/load_to_elasticsearch.py --wipe=false
```

![Upsert Data ro ElasticSearch](/app/images/read_me/ES_INGEST.png)

**3) Ingest to Qdrant**

Script: `scripts/load_to_qdrant.py`

```bash
export QDRANT_URL=http://localhost:6333
python scripts/load_to_qdrant.py
```

![Upsert Data to Qdrant](/app/images/read_me/QDRANT_INGEST.png)

> You can switch between your `small_seed` and `medium_seed` files by editing the `SOURCES` arrays in those two ingest scripts.

---

## 🧑‍⚕️ App Usage

* Choose **User Type** (Patient / Provider / Researcher)
* Ask a medical question
* See **citations** under the answer
* Give 👍/👎 **feedback** (stored in PostgreSQL)

![User Feedback Collection](/app/images/read_me/user_review_pic.png)

*Feedback is automatically saved to PostgreSQL DB for analytics and quality monitoring.*
![User Feedback Collection](/app/images/read_me/pg_db.png)

---

## 📊 Monitoring (Grafana)

ResidentRAG logs feedback into dockerized PostgreSQL DB, which is then used as the source data for visualization in Grafana.

**🎯 Auto-Provisioned Dashboard Features:**
1. 📊 User Satisfaction Distribution (Pie Chart)
2. 📈 Feedback Trends Over Time (Time Series)
3. 👥 User Type Engagement & Satisfaction (Bar Chart)
4. 🎯 Satisfaction Rate by User Type (Table)
5. ⚙️ Response Detail Preferences (Pie Chart)

**✨ Zero Setup Required:** The dashboard and PostgreSQL data source are automatically configured when you run `docker compose up`. Just navigate to [http://localhost:3010](http://localhost:3010) and log in with `admin/admin`.

![Grafana Dashboard](app/images/read_me/Grafana%20Dashboard.png)

---

## 🛠️ Agentic Tool Technologies (Hybrid Search Tool, Wikipedia Search Tool, PubMed Search Tool)

**Core Search Technologies:**
- [**Elasticsearch**](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html) - Powers lexical/BM25 search
- [**Qdrant**](https://qdrant.tech/documentation/) - Powers semantic vector search

**Hybrid Search Retriever:**
1. **Dual Retrieval**: Queries both Elasticsearch (keyword matching) and Qdrant (semantic similarity) simultaneously
2. **Reciprocal Rank Fusion (RRF)**: Combines and reranks results from both search engines
3. **Score Normalization**: Balances lexical and semantic relevance scores

**External APIs implemented in Tools:**
- [**PubMed API (Bio.Entrez)**](https://people.duke.edu/~ccc14/pcfb/biopython/BiopythonEntrez.html) - Powers the `pubmed_search` tool for accessing NCBI's biomedical literature
- [**Wikipedia API**](https://www.mediawiki.org/wiki/API:Action_API) - Powers the `wikipedia_search` tool for retrieving encyclopedic medical content

**Primary Dataset:**
- [**MedRAG Dataset**](https://huggingface.co/MedRAG) - Hugging Face medical corpus (PubMed abstracts, textbooks, Wikipedia articles)

---

## 📊 Evaluation

Evaluation tests are included at the bottom of these files (under `if __name__ == "__main__":` blocks):

* `app/llm/openai_client.py`
* `app/llm/query_rewriter.py`
* `app/llm/rag_utils.py`
* `app/search/es_search.py`
* `app/search/hybrid_search.py`
* `app/search/qdrant_search.py`
* `/app/evaluation/retrieval_eval.py`

These run retrieval metrics (Hit@k, Recall, MRR, MAP, nDCG) and query rewriting tests.

## 🔍 System Internals & Logging

Monitor the agent's decision-making process and search operations in real-time:

```bash
docker compose logs -f streamlit
```

![System Logs](/app/images/read_me/streamlit_logs.png)


**📸 Screenshot Placeholder:** *Terminal output of evaluation metrics*


```markdown
## 🔌 Model Context Protocol (MCP) Integration

A *[FastMCP server](https://gofastmcp.com/getting-started/welcome)* is fully implemented as a **proof of concept** for future extensibility, though currently commented out since the agent's primary function is focused on medical information retrieval. The *[Model Context Protocol](https://www.philschmid.de/mcp-introduction)* architecture enables *seamless tool orchestration* and *inter-agent communication*, positioning ResidentRAG for advanced capabilities like:

- 📧 *Automated email notifications* to healthcare teams
- 🏥 *Electronic health record (EHR) integration* for patient-specific queries  
- 🔬 *Multi-modal data retrieval* from imaging systems, lab databases, and clinical decision support tools
- 🤖 *Agent-to-agent collaboration* where ResidentRAG could consult specialized medical AI agents (radiology AI, pharmacology expert, etc.)
- ⚡ *Real-time clinical workflow integration* through hospital information systems

The MCP server foundation is ready to unlock these *enterprise-grade medical AI capabilities* as the system evolves beyond its current search-focused implementation. This architectural decision ensures ResidentRAG can scale from a research tool to a comprehensive *clinical decision support platform*.
```

---

## 📄 License

MIT
