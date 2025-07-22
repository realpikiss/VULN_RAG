# migrate_kb3_code_hnsw.py

import os
import json
import logging
import numpy as np
from pathlib import Path
import hnswlib
from typing import Dict, List
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
def load_env():
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

load_env()

# === Configuration ===
KB1_JSON_PATH = Path(os.getenv("KB1_PATH"))
OUTPUT_INDEX_PATH = Path(os.getenv("KB3_INDEX_PATH"))
OUTPUT_METADATA_PATH = Path(os.getenv("KB3_METADATA_PATH"))
EMBEDDING_MODEL = os.getenv("KB3_MODEL")

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
    
    # Normalisation L2
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    dim = vectors.shape[1]
    logger.info(f"Embedding dimension: {dim}")

    # Créer l'index HNSW
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
    index.add_items(vectors)
    index.set_ef(50)  # Paramètre de recherche

    # Create directories if necessary
    OUTPUT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    index.save_index(str(OUTPUT_INDEX_PATH))
    with open(OUTPUT_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"HNSW index saved to {OUTPUT_INDEX_PATH}")
    logger.info(f"Metadata saved to {OUTPUT_METADATA_PATH}")



if __name__ == "__main__":
    migrate_kb3_code()
