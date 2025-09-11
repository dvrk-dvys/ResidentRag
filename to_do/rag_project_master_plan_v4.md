# 🧠🏥 Medical RAG Assistant - Updated Implementation Plan v4
**LLM Zoomcamp Final Project - Target: 18+ Points! 🎯**

---

## 🚀 **Current Status Update (Sept 4, 2025)**

### ✅ **Day 1 COMPLETED**
- 🐳 **Docker Infrastructure** → Docker Compose with Elasticsearch, Qdrant, Grafana running
- 📦 **Data Indexing** → Medical data successfully indexed in both Elasticsearch and Qdrant
- 🎯 **Qdrant Vector Search** → Pure semantic search implemented (`src/search/qdrant_search.py`)
- 🔤 **Elasticsearch BM25** → Pure text search implemented (`src/search/es_search.py`)  
- 🔀 **Hybrid Search RRF** → Qdrant+ES combination with Reciprocal Rank Fusion (`src/search/hybrid_search.py`)
- 🤖 **Basic RAG Pipeline** → OpenAI integration with all 3 search methods working

### 🔄 **Day 1 In Progress**
- 📊 **Hybrid Search Evaluation** → Currently finishing evaluation of all 3 search methods
- 📈 **Performance Metrics** → Hit Rate and MRR comparisons (hybrid expected to win)

---

## 📅 **Day 2: LangChain Integration & Advanced RAG** 
*Goal: Add LangChain orchestration + auto-expansion capabilities*

### 🧩 **Morning (8-12pm): LangChain Integration (NEW)**
**Based on your LangChain notes - implementing the "glue layer" approach:**

- 🔗 **Wrap Hybrid Search as LangChain Retriever**
  - **Tools:** `langchain-core`, `langchain-community` 
  - **Code:** `src/search/langchain_hybrid_retriever.py`
  - **Feature:** Convert existing `hybrid_search.py` to LangChain `BaseRetriever` interface
  - **Benefit:** Standardized Document interface, easy composition

**LangChain Setup Instructions:**
```python
# For evaluation (fast, minimal) - stick with es_client.search(...)
hybrid_results = es_client.search(index=index_name, body=hybrid_query)
result_ids = [hit['_source']['id'] for hit in hybrid_results['hits']['hits']]

# For RAG pipeline (Documents + chaining) - use ElasticsearchRetriever  
hybrid_retriever = ElasticsearchRetriever.from_es_params(
    index_name=index_name,
    body_func=hybrid_query,
    content_field='text',
    url=es_url,
)

hybrid_results = hybrid_retriever.invoke(query)
result_docs = []
for hit in hybrid_results:
    result_docs.append(hit.metadata['_source'])
return result_docs
```

**Trade-offs:**
- **Direct es_client.search(...)** → Fast, minimal for metrics (Hit@k/MRR) 
- **ElasticsearchRetriever** → Document objects for RAG chains, more overhead

- 🎭 **RAG Chain Composition**
  - **Tools:** `ChatPromptTemplate`, `RunnableLambda`, `RunnablePassthrough`
  - **Code:** `src/llm/langchain_rag_pipeline.py`
  - **Feature:** Clean retrieval → context formatting → LLM chain
  - **Replaces:** Current basic RAG pipeline with composable LangChain version

### 🔄 **Afternoon (1-5pm): Auto-Expansion System (NEW)**
**Implementing the "can't answer → fetch → retry" loop from your notes:**

- 🤔 **Smart Routing Logic**
  - **Tools:** Retrieval confidence scoring, `RunnableBranch`
  - **Code:** `src/llm/expansion_router.py`
  - **Logic:** Check RRF scores + context length → decide if KB expansion needed
  - **Thresholds:** `max_rrf < 0.01` OR `total_chars < 500` triggers expansion

- 📚 **Knowledge Base Auto-Expansion**
  - **Tools:** `WikipediaLoader`, `RecursiveCharacterTextSplitter`
  - **Code:** `src/ingestion/auto_expand_kb.py`  
  - **Flow:** Query → Wikipedia fetch → chunk → upsert to Qdrant+ES → retry search
  - **Benefit:** Automatic knowledge enrichment when existing KB insufficient

- 🔁 **Retry Pipeline**
  - **Tools:** LangChain routing, your existing hybrid search
  - **Code:** `src/llm/auto_retry_rag.py`
  - **Flow:** 
    1. Try hybrid retrieval 
    2. If low confidence → expand KB with Wikipedia
    3. Retry hybrid search on enriched KB
    4. Answer with enhanced context

### 📊 **Evening (6-9pm): LangChain Evaluation**
- 🆚 **Compare Traditional vs LangChain RAG**
  - **Baseline:** Your current hybrid RAG pipeline  
  - **Enhanced:** LangChain version with auto-expansion
  - **Metrics:** Response quality, knowledge coverage, expansion frequency
- 📈 **Auto-Expansion Effectiveness**
  - **Test:** Questions that trigger Wikipedia expansion
  - **Measure:** Before/after expansion answer quality
  - **Document:** When expansion helps vs when it doesn't

---

## 📅 **Day 2 Continued: Interface & Advanced Features**

### 🎨 **LangChain-Enhanced Streamlit Interface**
- 💬 **Enhanced Chat with Auto-Expansion Visibility**
  - **Feature:** Show when Wikipedia expansion triggered
  - **Tools:** Streamlit + LangChain callbacks
  - **Code:** Update `app/main.py` to show expansion events

- 🎛️ **Advanced Search Controls**
  - **Feature:** Toggle auto-expansion on/off
  - **Feature:** Choose expansion sources (Wikipedia, HuggingFace, etc.)
  - **Tools:** Streamlit widgets, LangChain tool selection

---

## 📅 **Day 3: Monitoring & Production (UNCHANGED)**
*Goal: Full monitoring + production-ready*

### **Morning (8-12pm): Monitoring Dashboard (2 POINTS)**
- 📊 **Enhanced Metrics with LangChain**
  - **Additional Metrics:** Expansion trigger rate, Wikipedia fetch times
  - **LangChain:** Use callbacks for tracing retrieval → expansion → retry flows
  - **Tools:** `prometheus-client`, LangChain observability

### **Afternoon (1-5pm): Documentation (2 POINTS)**  
- 📖 **Updated Documentation**
  - **New Section:** LangChain integration benefits and architecture
  - **New Section:** Auto-expansion system explanation  
  - **Updated:** Setup instructions with LangChain dependencies

---

## 🔗 **LangChain Integration Benefits (From Your Notes)**

### ✅ **What LangChain Provides:**
- **Standard Interfaces:** `BaseRetriever`, `Document` - your hybrid search plugs in anywhere
- **Clean Composition:** Retrieval → routing → expansion → retry chains
- **Built-in Tools:** Wikipedia loader, text splitters for KB expansion  
- **Observability:** Callbacks and tracing for monitoring expansion events
- **Flexibility:** Easy to add new expansion sources (HuggingFace, PubMed, etc.)

### 🎯 **Your Specific Implementation:**
**Single LLM + Two Tools approach:**
1. **HybridRRFRetriever Tool** → Your proven Qdrant+ES+RRF combination
2. **Expand-KB Tool** → Wikipedia → chunk → upsert → retry

**Flow:**
```
Query → Hybrid Search → Check Confidence → 
  IF Low: Wikipedia Expansion → Re-search → Answer
  IF High: Direct Answer
```

---

## 🎯 **Updated Scoring Strategy (18+ Points)**

### ✅ **Core Requirements (Already Completed/In Progress)**
1. **Problem Description (2pts)** → ✅ Medical RAG with hybrid search
2. **Retrieval Flow (2pts)** → ✅ Elasticsearch + Qdrant + RRF implemented  
3. **Retrieval Evaluation (2pts)** → 🔄 Currently finishing hybrid vs pure comparisons
4. **LLM Evaluation (2pts)** → ✅ Multiple prompts + models tested
5. **Interface (2pts)** → ✅ Streamlit chat with feedback (will enhance with LangChain)
6. **Ingestion (2pts)** → ✅ Automated medical data indexing
7. **Monitoring (2pts)** → ✅ Prometheus + Grafana ready (will add LangChain metrics)
8. **Containerization (2pts)** → ✅ Docker Compose with all services
9. **Reproducibility (2pts)** → ✅ Clear setup (will update for LangChain)

### 🌟 **Enhanced Bonus Points with LangChain**
- **Hybrid Search (1pt)** → ✅ Already implemented and evaluated
- **Query Enhancement (1pt)** → 🆕 LangChain auto-expansion system  
- **Advanced RAG (1pt)** → 🆕 LangChain orchestration with retry logic
- **Knowledge Expansion (1pt)** → 🆕 Automatic Wikipedia integration
- **Production Features (1pt)** → 🆕 LangChain observability and callbacks

---

## 🔧 **Technical Architecture Update**

### **Before LangChain (Day 1 - Completed):**
```
Query → [Qdrant Search | ES Search | Hybrid RRF] → Context → OpenAI → Answer
```

### **After LangChain (Day 2 - New):**
```
Query → LangChain HybridRetriever → Confidence Check → 
  Branch A: Direct Answer (high confidence)
  Branch B: Wikipedia Expansion → Re-retrieve → Answer (low confidence)
```

### **Key Files Added:**
- `src/search/langchain_hybrid_retriever.py` - LangChain wrapper for your hybrid search
- `src/llm/langchain_rag_pipeline.py` - Composable RAG chain  
- `src/ingestion/auto_expand_kb.py` - Wikipedia expansion system
- `src/llm/expansion_router.py` - Confidence-based routing logic
- `src/llm/auto_retry_rag.py` - Complete retry pipeline

---

## 🎉 **Success Criteria Updated**

### **Day 1 ✅ (COMPLETED)**
- [x] 3 search methods implemented and working
- [x] Medical data indexed in both ES and Qdrant  
- [x] Basic RAG pipeline functional
- [🔄] Hybrid search evaluation in progress

### **Day 2 🆕 (LangChain Focus)**
- [ ] LangChain hybrid retriever wrapper working
- [ ] Auto-expansion system implemented and tested
- [ ] Retry pipeline functional with Wikipedia integration  
- [ ] Enhanced Streamlit interface with expansion visibility
- [ ] Comparison of traditional vs LangChain RAG approaches

### **Day 3 ✅ (Enhanced Monitoring)**
- [ ] Grafana dashboard with LangChain metrics
- [ ] Documentation updated with LangChain architecture
- [ ] 20+ points achieved with advanced features

**🏆 Updated Target Score: 22+ points with LangChain auto-expansion system!**

---

## 🧠 **Key Implementation Notes from Your LangChain Research**

1. **Keep It Simple:** LangChain as "glue layer" - your hybrid search stays the same, just wrapped
2. **Single LLM Approach:** One model with two tools (retriever + expander) rather than multiple LLMs  
3. **Confidence-Based Routing:** Use retrieval scores and context length to trigger expansion
4. **Modular Design:** Each piece (retriever, expander, router) works independently
5. **Production Ready:** LangChain provides observability and standardization for production use

This v4 plan positions LangChain as the orchestration layer that makes your already-working hybrid search more intelligent and self-improving through automatic knowledge expansion.