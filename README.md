# 🛡️ **VulRAG: Hybrid Vulnerability Detection & Patch Generation System**

**A state-of-the-art RAG (Retrieval-Augmented Generation) system for C/C++ vulnerability detection and automated patch generation using multi-modal search and LLM integration.**

---

## 🎯 **Overview**

VulRAG combines multiple search modalities to provide contextual vulnerability analysis:

- **🔍 Textual Search** (Whoosh): Semantic matching on vulnerability descriptions
- **🧠 Structural Search** (FAISS): Code Property Graph (CPG) similarity matching  
- **💻 Code Search** (FAISS): Direct source code similarity matching
- **🔀 Intelligent Fusion** (RRF): Reciprocal Rank Fusion for optimal result ranking
- **🤖 LLM Integration** (Qwen): Context-aware detection and patch generation

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Code    │    │   Preprocessing  │    │  Multi-Search   │
│   (C/C++)      │───▶│  - Keywords      │───▶│  - KB1 (Whoosh) │
│                 │    │  - CPG Vector    │    │  - KB2 (FAISS)  │
│                 │    │  - Code Vector   │    │  - KB3 (FAISS)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  LLM Analysis   │    │  Context Build   │    │   RRF Fusion    │
│  - Detection    │◀───│  - Top Docs      │◀───│  - Score Merge  │
│  - Patch Gen    │    │  - Templates     │    │  - Ranking      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Knowledge Bases**

| KB | Technology | Content | Purpose |
|----|------------|---------|---------|
| **KB1** | Whoosh (Enriched) | Complete vulnerability documents | Textual semantic search |
| **KB2** | FAISS | CPG structural embeddings | Code structure similarity |
| **KB3** | FAISS | Raw code embeddings | Direct code similarity |

---

## 📁 **Project Structure**

```
VulRAG-Hybrid-System/
├── core/                           # Core system modules
│   ├── preprocessing/              # Input processing pipeline
│   │   ├── query_builder.py       # LLM keyword extraction
│   │   ├── cpg_embedder.py        # CPG → vector generation
│   │   └── vulrag_preprocessor.py # Main preprocessing orchestrator
│   ├── retrieval/                  # Multi-modal search system
│   │   ├── rrf_fusion.py          # Reciprocal Rank Fusion
│   │   └── controller.py          # Search orchestration
│   ├── generation/                 # LLM integration
│   │   ├── context_builder.py     # Document → LLM prompt
│   │   ├── detector.py            # Vulnerability detection
│   │   └── patcher.py             # Patch generation
│   ├── evaluation/                 # Quality metrics
│   └── pipeline.py                # Main system entry point
├── scripts/                        # Standalone utilities
│   ├── migration/                  # KB creation scripts
│   │   ├── migrate_kb1_to_whoosh.py
│   │   ├── migrate_kb2_to_faiss.py
│   │   └── migrate_kb3_code_faiss.py
│   └── retrieval/                  # Individual search modules
│       ├── search_kb1.py          # Whoosh textual search
│       ├── search_kb2_faiss.py    # FAISS CPG search
│       └── search_kb3_code_faiss.py # FAISS code search
├── data/                           # Knowledge bases & datasets
│   ├── KBs/                        # Generated knowledge bases
│   │   ├── JSON_FORMAT_KB/         # Source data (kb1.json, kb2.json)
│   │   ├── kb1_index/              # Whoosh enriched index
│   │   ├── kb2_index/              # FAISS CPG index + metadata
│   │   └── kb3_index/              # FAISS code index + metadata
│   └── Dataset/                    # Evaluation datasets
└── tests/                          # Test suites
```

---

## 🚀 **Installation & Setup**

### **Requirements**

```bash
# Core dependencies
pip install whoosh faiss-cpu sentence-transformers
pip install numpy pandas scikit-learn

# LLM integration (choose one)
pip install ollama-python  # For local Qwen
# OR
pip install openai         # For OpenAI API
```

### **Knowledge Base Setup**

1. **Prepare source data**:
   ```bash
   export KB1_PATH="data/KBs/JSON_FORMAT_KB/kb1.json"
   export KB2_PATH="data/KBs/JSON_FORMAT_KB/kb2.json"
   ```

2. **Generate knowledge bases**:
   ```bash
   # Create Whoosh enriched index (KB1)
   python scripts/migration/migrate_kb1_to_whoosh.py
   
   # Create FAISS CPG index (KB2)  
   python scripts/migration/migrate_kb2_to_faiss.py
   
   # Create FAISS code index (KB3)
   python scripts/migration/migrate_kb3_code_faiss.py
   ```

3. **Verify setup**:
   ```bash
   python tests/test_kb1_search.py
   python tests/test_kb2_search.py
   python tests/test_kb3_search.py
   ```

---

## 💻 **Usage**

### **Basic Vulnerability Analysis**

```python
from core.pipeline import VulRAGPipeline

# Initialize system
vulrag = VulRAGPipeline()

# Analyze vulnerable code
vulnerable_code = """
char buffer[10];
strcpy(buffer, user_input);  // Buffer overflow vulnerability
printf("Data: %s", buffer);
"""

# Get analysis results
result = vulrag.analyze(vulnerable_code)

print(f"Vulnerable: {result['is_vulnerable']}")
print(f"CWE: {result['cwe']}")
print(f"Explanation: {result['explanation']}")

if result['is_vulnerable']:
    patch = vulrag.generate_patch(vulnerable_code, result)
    print(f"Suggested patch:\n{patch}")
```

### **Advanced Search Configuration**

```python
# Custom search weights for RRF fusion
vulrag = VulRAGPipeline(
    rrf_weights={
        "kb1": 0.3,  # Textual search weight
        "kb2": 0.4,  # Structural search weight  
        "kb3": 0.3   # Code search weight
    },
    top_k=10
)

# Manual retrieval for research
documents = vulrag.retrieve_similar(
    code=vulnerable_code,
    include_scores=True,
    include_provenance=True
)
```

---

## 🔬 **Methodology**

### **Multi-Modal Retrieval**

1. **Preprocessing**: 
   - Extract semantic keywords using LLM (1 optimized call)
   - Generate CPG structural embeddings via Joern
   - Create direct code embeddings via SentenceTransformers

2. **Parallel Search**:
   - **KB1**: BM25 textual search on vulnerability descriptions
   - **KB2**: Cosine similarity search on CPG structure vectors
   - **KB3**: Cosine similarity search on raw code vectors

3. **Fusion**: Reciprocal Rank Fusion (RRF) combines rankings:
   ```
   score = Σ(weight_i / (k + rank_i)) for each KB_i
   ```

4. **Context Assembly**: Top documents → structured LLM prompt

### **Quality Assurance**

- **Supervision Bias Avoidance**: Separated descriptive vs. diagnostic content
- **Source Diversity**: Multiple similarity measures for robust matching
- **Performance Optimization**: Single enriched index vs. multiple JSON lookups

---

## 📊 **Current Status**

### **✅ Completed (~ 70%)**

- ✅ **Core Infrastructure**: All KBs created and indexed
- ✅ **Individual Searchers**: KB1, KB2, KB3 functional
- ✅ **Enriched Architecture**: Whoosh contains complete documents
- ✅ **Migration Scripts**: Automated KB generation
- ✅ **Test Framework**: Individual component validation

### **🔧 In Progress (~ 20%)**

- 🔧 **RRF Fusion Module**: Multi-modal result combination
- 🔧 **Retrieval Controller**: Parallel search orchestration
- 🔧 **Preprocessing Pipeline**: Automated query generation

### **📋 Planned (~ 10%)**

- 📋 **LLM Integration**: Detection and patch generation
- 📋 **Evaluation Metrics**: Precision, recall, patch quality
- 📋 **Performance Optimization**: Caching and batch processing

---

## 🧪 **Testing**

```bash
# Test individual components
python tests/test_kb1_search.py
python tests/test_kb2_search.py  
python tests/test_kb3_search.py

# Test end-to-end pipeline (when complete)
python tests/test_pipeline_complete.py

# Benchmark performance
python evaluation/benchmark_retrieval.py
```

---

## 📈 **Evaluation Datasets**

- **DiverseVul**: Large-scale vulnerability dataset
- **PrimeVul**: Curated high-quality test cases
- **Custom Test Suite**: Targeted vulnerability patterns

---

## 🤝 **Contributing**

### **Development Priorities**

1. **RRF Fusion Implementation** (High Priority)
2. **Preprocessing Pipeline Completion** (High Priority)  
3. **LLM Integration** (Medium Priority)
4. **Performance Optimization** (Low Priority)

### **Code Standards**

- **Modular Design**: Each component independently testable
- **Type Hints**: Full typing for maintainability
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit tests for all core functions

---

## 📝 **Research & Publications**

This system implements novel approaches in:
- **Multi-modal vulnerability retrieval**
- **Code Property Graph embeddings for security**
- **Hybrid RAG for code analysis**
- **Supervision bias mitigation in security datasets**

---

## ⚖️ **License**

[Your chosen license]

---

## 📧 **Contact**

[Your contact information]

---

**VulRAG represents the cutting edge of AI-assisted vulnerability analysis, combining the power of modern retrieval techniques with large language models for practical cybersecurity applications.**