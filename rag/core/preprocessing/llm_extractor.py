import hashlib
import re
from typing import Tuple
import ollama


class LLMExtractor:
    def __init__(self, model: str = "kirito1/qwen3-coder:latest"):
        self.model = model
        self.cache = {} # Cache pour éviter les doublons
        try:
            ollama.show(model)
        except Exception as e:
            print(f"Model check failed: {e}")

    def extract(self, code: str) -> Tuple[str, str]:
        """
        Extract purpose and functions in NATURAL format to match your Whoosh index.
        Your index contains natural sentences, not stemmed keywords!
        """
        code_hash = hashlib.md5(code.encode()).hexdigest()
        if code_hash in self.cache:
            return self.cache[code_hash]

        # Generate in the SAME natural format as your existing index
        prompt = f"""Analyze this code snippet:

{code}

1. What is the purpose of the function in the above code snippet? 
Please summarize in one sentence with the format:
"Function purpose: [your answer]"

2. Please summarize the functions of the above code snippet in list format:
"The functions of the code snippet are:
1. [function 1]
2. [function 2] 
3. [function 3]"

Be concise but use natural language like the examples in the database."""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 200},
            )
            content = response.get("response", "").strip()
        except Exception:
            return "", ""

        # Parse the natural responses (same format as your index)
        purpose_match = re.search(r"Function purpose:\s*(.+)", content, re.IGNORECASE)
        
        # Extract function list and join naturally
        function_lines = re.findall(r"\d+\.\s*(.+)", content)
        functions_text = " ".join(function_lines) if function_lines else ""

        purpose = purpose_match.group(1).strip() if purpose_match else ""
        functions = functions_text.strip()

        # Store in cache and return natural text (Whoosh will handle stemming during search)
        self.cache[code_hash] = (purpose, functions)
        return purpose, functions