from pathlib import Path
import json
import os
import numpy as np
import hnswlib
import logging
from pathlib import Path    
# === Configuration ===
KB2_JSON_PATH = os.getenv("KB2_PATH")
OUTPUT_INDEX_PATH = os.getenv("KB2_INDEX_PATH")
OUTPUT_METADATA_PATH = os.getenv("KB2_METADATA_PATH")

# === Logger setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KB2-Migration")

def migrate_kb2():
    """
    Migrate a KB2 JSON file to a HNSW index + metadata file
    """
    if not Path(KB2_JSON_PATH).exists():
        raise FileNotFoundError(f"Error: KB2 file not found: {KB2_JSON_PATH}")

    logger.info(f"Loading KB2 from {KB2_JSON_PATH}")
    with open(KB2_JSON_PATH, "r", encoding="utf-8") as f:
        kb2_data = json.load(f)

    vectors = []  # list of vectors
    metadata = []  # list of metadata entries

    for key, entry in kb2_data.items():
        vec = entry.get("embedding", [])
        if not vec or len(vec) < 10:  # safety filter
            logger.warning(f"Warning: empty or invalid embedding for {key}")
            continue
        try:
            vec = np.array(vec, dtype=np.float32)
            if np.isnan(vec).any():
                logger.warning(f"Error: NaN embedding for {key}, ignored")
                continue
            vectors.append(vec)
            metadata.append({
                "key": key,
                "cwe": key.split("_")[0]
            })
        except Exception as e:
            logger.error(f"Error: key {key} : {e}")

    # Normalization + indexing
    vector_array = np.vstack(vectors)  # concatenate the vectors
    # Normalisation L2
    vector_array = vector_array / np.linalg.norm(vector_array, axis=1, keepdims=True)

    dim = vector_array.shape[1]  # dimensionality of the vectors
    logger.info(f"Creating HNSW index with dimension {dim}")
    
    # Créer l'index HNSW
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=len(vector_array), ef_construction=200, M=16)
    index.add_items(vector_array)
    index.set_ef(50)  # Paramètre de recherche

    # Output directories
    Path(OUTPUT_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Saving
    index.save_index(str(OUTPUT_INDEX_PATH))
    with open(str(OUTPUT_METADATA_PATH), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Success: HNSW index written to {OUTPUT_INDEX_PATH}")
    logger.info(f"Success: metadata JSON written to {OUTPUT_METADATA_PATH}")
    logger.info(f"Total indexed: {len(metadata)}")

if __name__ == "__main__":
    migrate_kb2()
