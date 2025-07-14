import json
import hashlib
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
try:
    from rag.generation.qwen_interface import QwenInterface
except Exception:  # noqa: E722
    # Lightweight stub for unit tests when QwenInterface or its deps are unavailable
    class QwenInterface:  # type: ignore
        def generate_response(self, prompt, max_tokens=64, temperature=0):
            return ""
        # Backwards-compat alias used in this file
        generate = generate_response

class VulRAGPreprocessor:
    """Preprocess code to extract purpose and behaviour using Qwen or stub."""
    def __init__(self):
        self.cache_file = Path("rag/data/cache/vulrag_preprocessing.json") 
        self.cache_file.parent.mkdir(exist_ok=True)
        self.cache = self._load_cache()
        self.llm = QwenInterface()

    def _call_llm(self, prompt: str) -> str:
        """Unified call to the underlying LLM regardless of interface variant."""
        if hasattr(self.llm, "generate_response"):
            return self.llm.generate_response(prompt, max_tokens=64, temperature=0)
        if hasattr(self.llm, "generate"):
            return self.llm.generate(prompt)
        return ""
    
    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            return json.load(open(self.cache_file))
        return {}
    
    def _save_cache(self):
        json.dump(self.cache, open(self.cache_file, 'w'), indent=2)
    
    def _hash_code(self, code: str) -> str:
        normalized = re.sub(r'\s+', ' ', code.strip())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    # ---------------- internal helpers ----------------
    def _call_llm(self, prompt: str) -> str:
        """Call underlying LLM interface irrespective of available method name."""
        if hasattr(self.llm, "generate_response"):
            return self.llm.generate_response(prompt, max_tokens=64, temperature=0)
        return self._call_llm(prompt)

    def process_input_function(self, code: str) -> Dict:
        code_hash = self._hash_code(code)
        if code_hash in self.cache:
            return self.cache[code_hash]
        
        # 2 appels Qwen parallèles
        with ThreadPoolExecutor(max_workers=2) as executor:
            purpose_future = executor.submit(self._extract_purpose, code)
            behavior_future = executor.submit(self._extract_behavior, code)
        
        result = {
            "raw_code": code,
            "raw_code": code,
            "purpose": purpose_future.result(),
            "behavior": behavior_future.result()
        }
        
        self.cache[code_hash] = result
        self._save_cache()
        return result
    
    def _extract_purpose(self, code: str) -> str:
        prompt = f"""
{code}

What is the purpose of the function in the above code snippet? 
Please summarize the answer in one sentence with the following format: 
"Function purpose: [your answer]"
"""
        return self._call_llm(prompt)
    
    def _extract_behavior(self, code: str) -> str:
        prompt = f"""
{code}

Please summarize the functions of the above code snippet in the list format without any other explanation:
"The functions of the code snippet are:
1. [function 1]
2. [function 2] 
3. [function 3]
..."
"""
        return self._call_llm(prompt)