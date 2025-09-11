# 🧠🏥 Medical RAG Assistant - Detailed Implementation Plan
**LLM Zoomcamp Final Project - Target: 18+ Points! 🎯**

---

## 🔧 **Tool Overview & Usage Guide**

### 🔍 **Search Technologies (Qdrant-First Strategy)**
- **🎯 Qdrant (PRIMARY):** Vector/semantic search with medical embeddings - YOUR FAVORITE! 
- **🔤 Elasticsearch (SECONDARY):** BM25 text search for exact medical terminology
- **🔀 Hybrid Search:** Qdrant-led combination with ES support for best results

### 🧠 **LLM & Evaluation**
- **🤖 OpenAI:** GPT-4o-mini (fast/cheap) vs GPT-4o (powerful/expensive)
- **📝 Multiple Prompts:** Basic Q&A, context-aware, chain-of-thought
- **📊 Evaluation:** Hit Rate, MRR, LLM response quality comparison

### 💾 **Data Storage**
- **🐘 PostgreSQL:** User feedback, metrics, application data
- **📁 JSON Files:** Medical seed data, processed documents

### 📊 **Monitoring Stack**
- **🎯 Prometheus:** Collect app metrics (response times, query counts)
- **📈 Grafana:** Dashboard with 5+ charts from Prometheus data

---

- **BM25** : classic keyword relevance. Great for exact terminology and phrases.

- **Dense vector (kNN)** : semantic similarity via embeddings.

- **Cosine + normalization** : you normalize vectors because cosine cares about direction, not length.

- **RRF (Reciprocal Rank Fusion)** : simple, strong way to merge ranked lists by ID.

- **source_type** : your lever for filtering/slicing results (wikipedia/textbook/pubmed).

- **title boost (title^2)** : makes matches in titles count more than body text.


## 🚀 **Day Zero Status (TODAY - Sept 2, 2025)**

### ✅ **COMPLETED** 
- 🎉 **MedRAG Dataset Successfully Downloaded** → `scripts/create_medical_seed.py`
- 🐳 **Docker Compose Infrastructure Setup** → `docker-compose.yml`
- 📦 **Dependencies Defined** → `requirements.txt`, `Pipfile`, `Pipfile.lock`
- 🔧 **Test Docker Stack** → `docker-compose up -d` → **Tools: Docker, curl**
- 🐍 **Create Virtual Environment** → `python -m venv venv` → **Tools: Python venv**
- 📁 **Create Project Directory Structure** → **Tools: mkdir, touch**

---

## 📅 **3-Day Sprint with Specific Tools**

### 🏗️ **Day 1: Foundation & Multi-Search Pipeline**
*Goal: 3 working search methods + basic RAG*

#### **Morning (9-12pm): Infrastructure & Data**
- 🐳 **Start Docker Stack** → `docker-compose up -d`
  - **Tools:** Docker Compose
- **Test:**
  - Elastic Search: http://localhost:9200/
  - QDrant: http://localhost:6333/dashboard
  - Grafana: http://localhost:3000/login

- 🔄 **Index Medical Data into Elasticsearch** 
  - **Tools:** `elasticsearch` Python client, `scripts/es_ingest.py`
  - **Code:** BM25 indexing with medical article fields
- 🎯 **Index Medical Data into Qdrant**
  - **Tools:** `qdrant-client`, `sentence-transformers`
  - **Code:** Generate embeddings, store vectors with metadata

#### **Afternoon (1-5pm): Triple Search Implementation (Qdrant-First!)** 
- 🎯 **Method 1: Pure Qdrant Vector Search (PRIMARY)**
  - **Tools:** `qdrant-client`, `sentence-transformers` 
  - **Code:** `src/search/qdrant_search.py` - medical semantic similarity
- 🔤 **Method 2: Pure Text Search (ES BM25)**
  - **Tools:** `elasticsearch` client
  - **Code:** `src/search/text_search.py` - medical terminology matching
- 🔀 **Method 3: Qdrant + ES Hybrid (WINNING COMBO)**  
  - **Tools:** Qdrant (primary) + ES (secondary), RRF combination
  - **Code:** `src/search/hybrid_search.py` - Qdrant-led hybrid with ES support

---
📝 Tiny README blurb (paste-able)

Why Qdrant doesn’t need BM25 here

We use Qdrant for dense, semantic retrieval and Elasticsearch for BM25 keyword retrieval.
Instead of duplicating BM25 in Qdrant, we run a hybrid at the app layer: Qdrant handles concept matching 
(e.g., “heart problems” → “cardiac issues”), while Elasticsearch catches exact medical terms, acronyms, and phrases 
(e.g., “MI”, “COPD”, “alpha-bisabolol”). We fuse both result lists with Reciprocal Rank Fusion (RRF),
which consistently outperforms either method alone. This keeps the stack simpler (one BM25 engine)
and maximizes retrieval quality.

✅ Yes, you should prove Hybrid > single retrievers

For your Retrieval Evaluation (2 pts), compare three modes with the same ground truth:
- qdrant (dense only)
- es_bm25 (BM25 only)
- hybrid (RRF of the two)

You already have wrappers; run Hit@k / MRR on all three and document the results. (You can skip ES kNN if you want — Qdrant covers dense.)

“so you can RRF-fuse with Elasticsearch later” — what that means

RRF (Reciprocal Rank Fusion) is a simple way to merge two ranked lists. You take top-N IDs from Qdrant and ES,
assign each ID a score like 1/(K+rank), sum scores across lists, and re-rank.
---



#### **Evening (6-9pm): RAG Pipeline**
- 🤖 **OpenAI Integration**
  - **Tools:** `openai` Python client
  - **Code:** `src/llm/openai_client.py`
- 📝 **Basic RAG Flow**
  - **Tools:** Search methods + OpenAI
  - **Code:** `src/llm/rag_pipeline.py` - search → context → LLM → answer
- ✅ **Test All 3 Search Methods**
  - **Tools:** Python scripts, medical questions
  - **Output:** 3 different search results for same query

### 🎨 **Day 2: Evaluation & Interface** 
*Goal: Compare all methods + web interface*

#### **Morning (8-12pm): Retrieval Evaluation (2 POINTS)**
- 📊 **Create Ground Truth Dataset**
  - **Tools:** Manual curation + GPT generation
  - **Code:** `src/evaluation/ground_truth.py` - 100+ Q&A pairs
- 📈 **Evaluate All 3 Search Methods**
  - **Tools:** Hit Rate, MRR calculations from course
  - **Code:** `src/evaluation/retrieval_eval.py`
  - **Compare:** Pure Qdrant vs Pure ES vs Qdrant+ES Hybrid
  - **Expected Winner:** Qdrant+ES Hybrid (92%+ Hit Rate vs 77% pure vector)
  - **Result:** Documentation of BEST performing method ← **REQUIRED FOR 2 POINTS**

#### **Afternoon (1-5pm): LLM Evaluation (2 POINTS)**  
- 🧠 **Test Multiple LLM Approaches**
  - **Tools:** OpenAI API (GPT-4o-mini vs GPT-4o)
  - **Code:** `src/llm/llm_comparison.py`
- 📝 **Test Multiple Prompt Strategies**
  - **Prompt 1:** Basic Q&A template
  - **Prompt 2:** Context-aware with medical focus  
  - **Prompt 3:** Chain-of-thought reasoning
  - **Tools:** `src/llm/prompt_templates.py`
  - **Result:** Documentation of BEST LLM + prompt combo ← **REQUIRED FOR 2 POINTS**

#### **Evening (6-10pm): Streamlit Interface (2 POINTS)**
- 💬 **Build Chat Interface** 
  - **Tools:** `streamlit`, session state
  - **Code:** `app/main.py` - chat history, message display
- 👍 **Add User Feedback System**
  - **Tools:** Streamlit widgets, PostgreSQL
  - **Code:** `src/monitoring/feedback_collector.py` - thumbs up/down, ratings
- 🎛️ **Add Search Method Selection**
  - **Tools:** Streamlit selectbox
  - **Feature:** User can choose Text/Vector/Hybrid search

### 📊 **Day 3: Monitoring & Documentation**
*Goal: Full monitoring + production-ready*

#### **Morning (8-12pm): Monitoring Dashboard (2 POINTS)**
- 🎯 **App Metrics Collection**
  - **Tools:** `prometheus-client` in Python app
  - **Code:** `src/monitoring/metrics_logger.py` - track query times, feedback scores
- 📈 **Grafana Dashboard with 5+ Charts**
  - **Tools:** Grafana UI, Prometheus data source
  - **Charts Required:**
    1. Query volume over time
    2. Response time distribution  
    3. User feedback scores (from PostgreSQL)
    4. Search method usage comparison
    5. Most popular medical topics
  - **Result:** Working dashboard ← **REQUIRED FOR 2 POINTS**

#### **Afternoon (1-5pm): Documentation (2 POINTS)**
- 📖 **Comprehensive README**
  - **Tools:** Markdown, screenshots
  - **Sections:** Setup, usage, evaluation results, architecture
- 🔄 **Reproducible Setup Instructions**
  - **Tools:** Step-by-step Docker commands
  - **Test:** Fresh machine setup verification
  - **Result:** Anyone can run your project ← **REQUIRED FOR 2 POINTS**

#### **Evening (6-9pm): Final Testing & Bonus**
- 🌟 **Bonus Features Implementation**
  - **Document Re-ranking (1pt):** Cross-encoder model after search
  - **Query Rewriting (1pt):** Improve user queries before search
  - **Hybrid Search (1pt):** Already implemented ✅
- 🌐 **Cloud Deployment (2pts):** Deploy to AWS/GCP if time permits

---

## 🎯 **Scoring Breakdown with Tools**

### ✅ **Core 18 Points - Tool Mapping**
1. **Problem Description (2pts)** → Markdown documentation
2. **Retrieval Flow (2pts)** → Elasticsearch + Qdrant + OpenAI integration  
3. **Retrieval Evaluation (2pts)** → Hit Rate/MRR comparison of 3 search methods
4. **LLM Evaluation (2pts)** → Multiple models + prompt testing
5. **Interface (2pts)** → Streamlit chat application
6. **Ingestion (2pts)** → ✅ MedRAG Python script (automated)
7. **Monitoring (2pts)** → Prometheus + Grafana + user feedback
8. **Containerization (2pts)** → ✅ Docker Compose with all services  
9. **Reproducibility (2pts)** → Clear README + working setup

### 🌟 **Bonus Points Strategy (5+ pts)**
- **Hybrid Search (1pt)** → ✅ Planned (ES + Qdrant combination)
- **Document Re-ranking (1pt)** → Cross-encoder after initial search
- **Query Rewriting (1pt)** → Query enhancement before search  
- **Cloud Deployment (2pts)** → AWS/GCP deployment
- **Extra Features (1pt)** → Advanced analytics, multi-language support

---

## 🔍 **When to Use Each Search Technology (Qdrant-First!)**

### 🎯 **Qdrant (PRIMARY - Your Favorite!)**  
**Use for:**
- **Medical concept understanding** → "Heart problems" finds "cardiac issues" 
- **Semantic similarity** → Understanding medical relationships
- **Primary search engine** → Leading the hybrid approach
- **Medical embeddings** → Specialized medical knowledge vectors

### 🔤 **Elasticsearch (SECONDARY Support)**
**Use for:**
- **Exact medical terminology** → "myocardial infarction" exact matches
- **Acronym searches** → "MI", "COPD", "BP" medical abbreviations  
- **Supporting Qdrant** → Complementing semantic search
- **Fallback search** → When vector search misses obvious keywords

### 🔀 **Qdrant+ES Hybrid (WINNING APPROACH)**
**Best for:**
- **Production medical system** → Qdrant leads, ES supports
- **Highest evaluation scores** → Expected 92%+ Hit Rate (course data shows this)
- **Bonus points** → Advanced Qdrant-centered RAG
- **Your final system** → Qdrant as primary with ES enhancement

---

## 📊 **Evaluation Strategy for Full Points**

### 🔍 **Retrieval Evaluation (2 points)**
**Must test ALL THREE approaches:**
1. **Pure Qdrant Vector** → Medical semantic similarity performance (expected ~77%)
2. **Pure ES Text** → Medical terminology matching performance  
3. **Qdrant+ES Hybrid** → Combined approach (expected winner ~92%+)

**Tools:** Hit Rate, MRR metrics → Document Qdrant+ES hybrid as winner → Use in final app

### 🧠 **LLM Evaluation (2 points)** 
**Must test MULTIPLE approaches:**
1. **Models:** GPT-4o-mini vs GPT-4o cost/performance
2. **Prompts:** Basic Q&A vs Context-aware vs Chain-of-thought
3. **Settings:** Temperature, max_tokens variations

**Result:** Document best LLM + prompt combination → Use in final app

---

## 🎉 **Success Checklist**

### **Day 1 ✅**
- [ ] 3 search methods implemented and working
- [ ] Medical data indexed in both ES and Qdrant  
- [ ] Basic RAG pipeline functional
- [ ] All search methods tested with same queries

### **Day 2 ✅**
- [ ] Ground truth dataset created (100+ medical Q&A)
- [ ] All 3 retrieval methods evaluated and compared
- [ ] Multiple LLM approaches tested and documented
- [ ] Streamlit interface with chat and feedback

### **Day 3 ✅**
- [ ] Grafana dashboard with 5+ monitoring charts
- [ ] User feedback collection working
- [ ] Comprehensive documentation completed
- [ ] 18+ points verified across all criteria

**🏆 Target Score: 21+ points with documented comparisons and best method selection!**