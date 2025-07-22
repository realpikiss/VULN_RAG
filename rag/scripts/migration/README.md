# Migration Scripts

This directory contains one–off utilities used to build or rebuild the various
search indexes that power the Retrieval-Augmented Generation (RAG) pipeline.
Each script converts a JSON knowledge-base dump into a high-performance search
index (Whoosh or HNSW) and stores extra metadata required by the search layer.

| Script | Purpose |
| ------ | ------- |
| **`migrate_kb1_to_whoosh.py`** | Reads the *KB1* vulnerability JSON, optionally enriches each entry with *KB2* CPG data, and builds a **Whoosh** full-text index containing more than 25 structured fields. The script automatically deletes any existing index at `KB1_INDEX_PATH`, recreates it with the enriched schema, and prints useful statistics. |
| **`migrate_kb2_to_hnsw.py`** | Converts the pre-computed embedding vectors in *KB2* into a **HNSW** index for lightning-fast vector similarity search and stores lightweight metadata alongside the index. |
| **`migrate_kb3_code_hnsw.py`** | Generates fresh code embeddings for *KB1* source-code snippets using **Sentence-Transformer** `all-MiniLM-L6-v2`, normalises them, and builds a **HNSW** index + metadata file that we call *KB3*. |

All migration scripts are idempotent: you can run them anytime to regenerate
a clean index. They respect the following environment variables so they can be
easily plugged into CI pipelines or notebooks:

```
KB1_PATH, KB2_PATH, KB1_INDEX_PATH,
KB2_INDEX_PATH/KB2_METADATA_PATH,
KB3_INDEX_PATH/KB3_METADATA_PATH, KB3_MODEL
```

> **Tip:** Regenerate indexes after changing the schema or whenever the source
> JSON data is updated.
