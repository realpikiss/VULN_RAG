# VulnRAG: Hybrid Vulnerability Detection & Patch Generation

A comprehensive RAG-based system for detecting vulnerabilities in C/C++ code and generating automated patches using advanced static analysis, heuristics, and AI.

## 🏗️ Architecture

### **Hybrid Pipeline**
- **Static Analysis**: Cppcheck, Clang-Tidy, Flawfinder
- **Advanced Heuristics**: Semgrep with custom rules
- **AI Arbitration**: Independent LLM analysis for complex cases
- **Automated Patches**: System generates validated and optimized fixes

### **Multi-LLM Support**
- **Qwen2.5-Coder**: Fast, efficient detection
- **Kirito/Qwen3-Coder**: High-accuracy analysis
- **Hugging Face Integration**: Native support for transformer models

### **Knowledge Bases**
| Base | Technology | Content | Usage |
|------|------------|---------|-------|
| **KB1** | Whoosh | Enriched vulnerability documents | Textual semantic search |
| **KB2** | HNSW | CPG graph embeddings | Structural similarity |
| **KB3** | HNSW | Raw code embeddings | Direct code similarity |

## 🚀 Quick Start

### **Installation**

```bash
# Clone repository
git clone <repository-url>
cd VulnRAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### **Knowledge Base Setup**

```bash
# Set environment variables
export KB1_INDEX_PATH="rag/data/KBs/kb1_index"
export KB2_INDEX_PATH="rag/data/KBs/kb2_index"
export KB3_INDEX_PATH="rag/data/KBs/kb3_index"

# Generate search indexes
python rag/scripts/migration/migrate_kb1_to_whoosh.py
python rag/scripts/migration/migrate_kb2_to_hnsw.py
python rag/scripts/migration/migrate_kb3_code_hnsw.py
```

### **Usage**

```bash
# Launch web interface
streamlit run app.py

# Or use Python API
python evaluation/detection/quick_test.py
```

## 📊 Knowledge Bases

The system uses 3 specialized knowledge bases:

| Base | Technology | Content | Usage |
|------|------------|---------|-------|
| **KB1** | Whoosh | Enriched vulnerability documents | Textual semantic search |
| **KB2** | HNSW | CPG graph embeddings | Structural similarity |
| **KB3** | HNSW | Raw code embeddings | Direct code similarity |

### **Characteristics**

- **KB1** : 25+ enriched fields (purpose, function, analysis, solution, code)
- **KB2** : 384D vectors based on Code Property Graphs
- **KB3** : On-the-fly embeddings with Sentence-Transformers

## 🔧 Advanced Features

### **Hybrid Detection**

- **Static Analysis** : Fast detection of critical vulnerabilities
- **Semgrep Heuristics** : Complex patterns and custom rules
- **LLM Arbitration** : Independent contextual decisions with explanations
