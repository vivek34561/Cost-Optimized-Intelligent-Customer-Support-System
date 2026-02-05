# Customer Support Chatbot

Intelligent customer support chatbot with 3-tier routing system (Zero-cost FAQ, RAG + Small LLM, Escalation) achieving **79.6% cost reduction** vs uniform big LLM approach.

## 🎯 Key Features

- **Intent Classification:** 97.69% accuracy on 27 intents using TF-IDF + Logistic Regression
- **Smart Routing:** 3-bucket system (Zero-cost FAQ, RAG, Escalation)
- **Local Embeddings:** HuggingFace sentence-transformers (free, runs offline)
- **Local Vector DB:** FAISS (no cloud service needed)
- **Fast LLM:** Groq (750+ tokens/sec)
- **LangGraph:** State machine orchestration
- **Confidence Fallback:** Low confidence queries routed safely to RAG

## 📊 Performance

- **Cost Savings:** 79.6% reduction
- **Zero-cost Routing:** 30.6% of queries
- **Low-cost Routing:** 51.6% of queries
- **High-cost Routing:** 17.8% of queries
- **Intent Accuracy:** 97.69%

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements-rag.txt
```

### 2. Set Up Groq API Key
Create `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key_here
```
Get free API key from: https://console.groq.com

### 3. Build FAISS Index
```bash
# Test with 100 documents
python build_rag_index.py --limit 100

# Or full index (26,872 documents)
python build_rag_index.py
```

### 4. Run Chatbot
```bash
# Test mode
python src/main.py

# Interactive mode
python src/main.py interactive
```

## 🏗️ Architecture

### Technology Stack
- **Intent Classification:** scikit-learn (TF-IDF + Logistic Regression)
- **Embeddings:** HuggingFace sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** FAISS (local, no cloud)
- **LLM:** Groq (`llama-3.3-70b-versatile`)
- **Orchestration:** LangGraph state machine
- **Dataset:** Bitext Customer Support (26,872 examples, 27 intents)

### Project Structure
```
src/
├── state/          # State definitions
├── llm/            # LLM configurations & prompts
├── nodes/          # LangGraph nodes (intent, retrieve, generate)
├── graph/          # State machine workflow
└── main.py         # Main chatbot interface

models/             # Intent classification models
data/               # FAISS index (created on build)
intent_router.py    # Reusable routing module
```

### Routing System

**BUCKET_A (Zero-cost):** FAQ intents with template responses
- No LLM needed
- 8 intents: check_invoice, track_order, payment_methods, etc.

**BUCKET_B (Low-cost):** RAG + Small LLM
- FAISS retrieval + Groq generation
- 15 intents: cancel_order, create_account, get_refund, etc.

**BUCKET_C (High-cost):** Escalation
- Complex issues requiring attention
- 4 intents: complaint, contact_human_agent, etc.

### LangGraph Flow
```
START → intent → [conditional] → generate → END
                      ↓
                  retrieve (only BUCKET_B) → generate → END
```

## 📚 Documentation

- **[RAG_SETUP.md](RAG_SETUP.md)** - Complete setup guide
- **[CONFIDENCE_FALLBACK.md](CONFIDENCE_FALLBACK.md)** - Confidence-based routing

## 🧪 Testing

### Test Intent Router
```bash
python intent_router.py
```

### Dry-Run Evaluation
```bash
python dry_run_evaluation.py
```

### Test RAG Retrieval
```bash
python src/retriever.py
```

## 💡 Why This Stack?

### HuggingFace Embeddings
- ✅ Free and runs locally
- ✅ No API calls or rate limits
- ✅ Fast inference (~1ms)
- ✅ Works offline after model download

### FAISS Vector Database
- ✅ No cloud service needed
- ✅ No API keys for vector DB
- ✅ Fast for small-medium datasets
- ✅ Can version control the index

### Groq LLM
- ✅ Ultra-fast inference (750+ tokens/sec)
- ✅ Free tier available
- ✅ Low latency
- ✅ Cost-effective

## 📦 Dependencies

```
# Core
pandas, numpy, scikit-learn

# Embeddings
sentence-transformers, torch

# Vector DB
faiss-cpu

# LLM
groq

# Orchestration
langchain, langchain-groq, langgraph
```

## 🔑 API Keys Needed

**Required:**
- Groq API Key (free at console.groq.com)

**Optional:**
- None! Embeddings and vector DB are local

## 📈 Cost Comparison

**Traditional Approach (uniform GPT-4):**
- 100% of queries → GPT-4 → High cost

**Our Approach:**
- 30.6% → Zero cost (templates)
- 51.6% → Low cost (Groq)
- 17.8% → High cost (escalation)
- **Result: 79.6% cost reduction**

## 🎓 Dataset

Bitext Customer Support Dataset v11
- **Size:** 26,872 instruction-response pairs
- **Intents:** 27 categories
- **Language:** English
- **Domain:** E-commerce customer support

## 🤝 Contributing

Feel free to open issues or submit PRs!

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- Bitext for the customer support dataset
- HuggingFace for sentence-transformers
- Meta for FAISS
- Groq for fast LLM inference
- LangChain team for LangGraph

---

**Ready to run?**
```bash
pip install -r requirements-rag.txt
python build_rag_index.py --limit 100
python src/main.py interactive
```
