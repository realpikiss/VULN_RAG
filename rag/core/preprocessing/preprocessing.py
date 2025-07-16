"""
VulRAG Preprocessing Pipeline 
======================================

complete pipeline : LLM + CPG in parallel -> format for fusion controller
"""

import hashlib
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np

@dataclass
class PreprocessingResult:
    """
    Preprocessing result: code, purpose, function, graph embedding, 
    processing time, cache hit
    """
    code: str
    purpose: str
    function: str  
    graph_embedding: np.ndarray
    processing_time_ms: float = 0.0
    cache_hit: bool = False
    
    def to_query_dict(self) -> Dict[str, Any]:
        """
        Convert result to query dictionary
        """
        return {
            "kb1_purpose": self.purpose,
            "kb1_function": self.function, 
            "kb2_vector": self.graph_embedding.tolist(),
            "kb3_code": self.code
        }

class PreprocessingPipeline:
    """
    Preprocessing pipeline: LLM + CPG in parallel
    """
    def __init__(self, 
                 llm_model: str = "kirito1/qwen3-coder:latest",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 cache_dir: str = "rag/data/cache/preprocessing"):
        """
        Initialize preprocessing pipeline
        """
        from .llm_extractor import LLMExtractor
        from .cpg_extractor import CPGExtractor
        
        self.llm_extractor = LLMExtractor(llm_model)
        self.cpg_extractor = CPGExtractor(embedding_model)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def process(self, code: str) -> PreprocessingResult:
        """
        Perform preprocessing on code
        """
        start_time = time.time()
        
        # Check cache
        cache_key = hashlib.sha256(code.encode()).hexdigest()[:16]
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
                result.cache_hit = True
                return result
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=2) as executor:
            llm_future = executor.submit(self.llm_extractor.extract, code)
            cpg_future = executor.submit(self.cpg_extractor.extract_embedding, code)
            
            purpose, function = llm_future.result()
            graph_embedding = cpg_future.result()
        
        result = PreprocessingResult(
            code=code,
            purpose=purpose,
            function=function,
            graph_embedding=graph_embedding,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # Cache save
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result

def create_pipeline(llm_model: str = "kirito1/qwen3-coder:latest",
                   embedding_model: str = "all-MiniLM-L6-v2") -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline
    """
    return PreprocessingPipeline(llm_model, embedding_model)