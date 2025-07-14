# migrate_kb3_code_faiss.py

import os
import json
import logging
import numpy as np
from pathlib import Path
import faiss
from typing import Dict, List
from sentence_transformers import SentenceTransformer

# === Configuration ===
KB1_JSON_PATH = Path(os.getenv("KB1_PATH", "data/processed/kb1_enriched/kb1.json"))
OUTPUT_INDEX_PATH = Path(os.getenv("KB3_INDEX_PATH", "data/indexes/kb3_index/kb3_code.index"))
OUTPUT_METADATA_PATH = Path(os.getenv("KB3_METADATA_PATH", "kb3_metadata.json"))
EMBEDDING_MODEL = os.getenv("KB3_MODEL", "all-MiniLM-L6-v2")

# === Logger ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KB3-Migration")


def migrate_kb3_code():
    if not KB1_JSON_PATH.exists():
        raise FileNotFoundError(f"KB1 file not found: {KB1_JSON_PATH}")

    logger.info(f"Loading KB1 from {KB1_JSON_PATH}")
    with open(KB1_JSON_PATH, 'r', encoding='utf-8') as f:
        kb1_data: Dict[str, Dict] = json.load(f)

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    codes = []
    metadata = []

    for key, entry in kb1_data.items():
        code = entry.get("code_before_change", "").strip()
        if not code:
            logger.warning(f"Missing code for {key}, entry ignored.")
            continue

        codes.append(code)
        metadata.append({
            "key": key,
            "cwe": key.split("_")[0]
        })

    logger.info(f"Generating embeddings for {len(codes)} code snippets...")
    vectors = model.encode(codes, batch_size=16, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    logger.info(f"Embedding dimension: {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Create directories if necessary
    OUTPUT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(OUTPUT_INDEX_PATH))
    with open(OUTPUT_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"FAISS index saved to {OUTPUT_INDEX_PATH}")
    logger.info(f"Metadata saved to {OUTPUT_METADATA_PATH}")
    logger.info(f"Total indexed: {len(metadata)}")


if __name__ == "__main__":
    migrate_kb3_code()
