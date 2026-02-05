# RAG Chatbot with LangGraph & FAISS

## 📁 Project Structure

```
Customer Support Chatbot/
├── src/
│   ├── config.py                      # Configuration
│   ├── faiss_index_builder.py         # Build FAISS index
│   ├── retriever.py                   # RAG retriever
│   ├── state/                         # State definitions
│   │   ├── __init__.py
│   │   └── state.py                   # ChatbotState TypedDict
│   ├── llm/                           # LLM configurations
│   │   ├── __init__.py
│   │   ├── models.py                  # Groq LLM setup
│   │   └── prompts.py                 # System prompts
│   ├── nodes/                         # LangGraph nodes
│   │   ├── __init__.py
│   │   ├── intent_node.py             # Intent classification
│   │   ├── retrieve_node.py           # RAG retrieval
│   │   └── generate_node.py           # LLM generation
│   ├── graph/                         # State machine
│   │   ├── __init__.py
│   │   └── chatbot_graph.py           # LangGraph workflow
│   └── main.py                        # Main chatbot interface
├── models/                            # Intent classification models
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression_model.pkl
│   └── routing_config.json
├── data/                              # FAISS index (created on build)
│   ├── faiss_index
│   └── faiss_metadata.json
├── intent_router.py                   # Intent routing module
├── build_rag_index.py                 # Script to build FAISS index
├── requirements-rag.txt               # RAG dependencies
└── .env                              # API keys (create from .env.example)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-rag.txt
```

**Key Dependencies:**
- `sentence-transformers` - HuggingFace embeddings (free, runs locally)
- `faiss-cpu` - Local vector database (no cloud service needed)
- `groq` - Fast LLM inference (free tier available)
- `langgraph` - State machine orchestration

### 2. Set Up API Keys

Create `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:
```env
# Get free API key from: https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Optional: Override default models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=llama-3.3-70b-versatile
```

### 3. Build FAISS Index

**Option A: Test with 100 documents (recommended first)**
```bash
python build_rag_index.py --limit 100
```

**Option B: Full index (26,872 documents)**
```bash
python build_rag_index.py
```

The embedding model (all-MiniLM-L6-v2, ~90MB) will download automatically on first run.

### 4. Test the Chatbot

**Run test queries:**
```bash
python src/main.py
```

**Interactive mode:**
```bash
python src/main.py interactive
```

## 🏗️ Architecture

### Embedding Model (HuggingFace)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension:** 384
- **Advantages:**
  - ✅ Free, runs locally
  - ✅ No API calls needed
  - ✅ Fast inference (~1ms per query)
  - ✅ Good quality for semantic search

### Vector Database (FAISS)
- **Type:** IndexFlatIP (cosine similarity via normalized vectors)
- **Storage:** Local files (`data/faiss_index` + `data/faiss_metadata.json`)
- **Advantages:**
  - ✅ No cloud service needed
  - ✅ No API keys for vector DB
  - ✅ Fast for small-medium datasets
  - ✅ Can version control the index

### LLM (Groq)
- **Model:** `llama-3.3-70b-versatile`
- **Advantages:**
  - ✅ Ultra-fast inference (up to 750 tokens/sec)
  - ✅ Free tier available
  - ✅ Low latency

### LangGraph State Machine

```
START → intent_node → [conditional routing] → generate_node → END
                              ↓
                        retrieve_node (only for BUCKET_B)
                              ↓
                        generate_node → END
```

**Flow:**
1. **intent_node**: Classifies intent and determines bucket (A/B/C)
2. **Conditional routing**: 
   - BUCKET_A (FAQ) → skip retrieval, go to generate
   - BUCKET_B (RAG) → retrieve from FAISS → generate
   - BUCKET_C (Escalation) → skip retrieval, go to generate
3. **retrieve_node**: Query FAISS index for relevant documents
4. **generate_node**: Generate response using Groq LLM with context

## 🎯 Three-Bucket Routing System

### BUCKET_A: Zero-Cost (Direct Responses)
**Intents:** check_invoice, check_payment_methods, track_order, delivery_options, check_refund_policy, check_cancellation_fee, delivery_period, track_refund

**Handling:** Template responses, no LLM needed

**Cost:** $0

### BUCKET_B: Low-Cost (RAG + Small LLM)
**Intents:** cancel_order, change_order, change_shipping_address, create_account, delete_account, edit_account, get_invoice, get_refund, newsletter_subscription, payment_issue, place_order, recover_password, registration_problems, review, set_up_shipping_address, switch_account

**Handling:** FAISS retrieval → Groq LLM generation

**Cost:** ~$0.0001 per query

### BUCKET_C: High-Cost (Escalation)
**Intents:** complaint, payment_issue (low confidence), contact_human_agent, contact_customer_service

**Handling:** Escalation message or GPT-4 equivalent

**Cost:** Variable

## 📊 Performance Metrics

From dry-run evaluation (500 samples):
- **BUCKET_A (Zero-cost):** 30.6%
- **BUCKET_B (Low-cost):** 51.6%
- **BUCKET_C (High-cost):** 17.8%
- **Cost Savings:** 79.6% vs uniform big LLM approach

## 🔧 Configuration

### Embedding Configuration (src/config.py)
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
```

### RAG Configuration
```python
TOP_K_RETRIEVAL = 3  # Number of documents to retrieve
```

### LLM Configuration
```python
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 500
```

## 🧪 Testing Components

### Test Retrieval
```bash
python src/retriever.py
```

### Test Full Graph
```bash
python src/main.py
```

### Interactive Chat
```bash
python src/main.py interactive
```

Example session:
```
🧑 You: How do I track my order?
📊 Routing: BUCKET_A | track_order (98%)
🤖 Bot: I'd be happy to help you track your order! Please provide your order number...

🧑 You: How can I cancel my subscription?
📊 Routing: BUCKET_B | cancel_order (95%)
🤖 Bot: [RAG-powered response based on knowledge base]

🧑 You: I'm very unhappy with your service!
📊 Routing: BUCKET_C | complaint (92%)
🤖 Bot: I understand you're experiencing an issue...
```

## 🚀 Next Steps

1. **FastAPI Deployment:** Create REST API endpoint
2. **Conversation Memory:** Add chat history tracking
3. **Streaming Responses:** Implement real-time streaming
4. **Monitoring:** Add logging and analytics
5. **Production Deployment:** Docker containerization

## 💡 Tips

1. **First Run:** Start with `--limit 100` to test quickly
2. **Embedding Model:** Downloads once, cached thereafter (~90MB)
3. **FAISS Index:** Stored locally, can be committed to git if desired
4. **Groq API:** Free tier provides generous limits for testing
5. **No Internet:** Embeddings work offline after model download

## 🐛 Troubleshooting

**FAISS index not found:**
```bash
python build_rag_index.py --limit 10
```

**Embedding model slow to download:**
- Model downloads from HuggingFace Hub (~90MB)
- Subsequent runs use cached model

**Groq API errors:**
- Check API key in `.env`
- Verify API quota at console.groq.com

**Import errors:**
```bash
pip install -r requirements-rag.txt --upgrade
```

## 📝 License

See LICENSE file for details.
