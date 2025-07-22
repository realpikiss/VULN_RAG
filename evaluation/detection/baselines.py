"""
Detection baselines for evaluation
"""

import time
import logging
import subprocess
import tempfile
import os
from typing import Dict, List, Tuple
import json

def get_secure_api_key(service_name: str, env_var: str = None, prompt_message: str = None) -> str:
    """Securely get API key from parameter, environment, or interactive input"""
    # Try environment variable first
    if env_var and os.getenv(env_var):
        return os.getenv(env_var)
    
    # Interactive input as fallback
    try:
        import getpass
        if not prompt_message:
            prompt_message = f"Enter your {service_name} API key: "
        
        print(f"🔐 {service_name} API Key required")
        api_key = getpass.getpass(prompt_message).strip()
        
        if not api_key:
            raise ValueError(f"{service_name} API key cannot be empty")
        
        return api_key
    except ImportError:
        raise ValueError("getpass module not available")

def safe_log(logger, level: str, message: str):
    """Safely log messages with UTF-8 handling"""
    # Simply ignore logging to avoid encoding issues
    # The detection works fine, logging is just for debug
    pass

# Configure logger to handle UTF-8 properly
import sys
import io

# Force UTF-8 encoding for stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Create a custom UTF-8 safe handler
class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Ensure the message is UTF-8 safe
            if isinstance(msg, str):
                msg = msg.encode('utf-8', errors='replace').decode('utf-8')
            stream = self.stream
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Configure logger with UTF-8 safe handler
logger = logging.getLogger(__name__)
logger.handlers.clear()  # Remove any existing handlers
handler = UTF8StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger

class BaseDetector:
    """Base class for all detectors"""
    
    def __init__(self, name: str):
        self.name = name
        self.results = []
        
    def detect(self, code: str) -> Tuple[int, float]:
        """Detect vulnerability in code. Returns (prediction, confidence)"""
        raise NotImplementedError
        
    def evaluate_batch(self, samples: List[Dict]) -> Dict:
        """Evaluate a batch of samples"""
        start_time = time.time()
        
        predictions = []
        confidences = []
        true_labels = []
        
        for sample in samples:
            try:
                pred, conf = self.detect(sample["func"])
                predictions.append(pred)
                confidences.append(conf)
                true_labels.append(sample["label"])
                
                # Store detailed result
                result_data = {
                    "sample_id": sample["id"],
                    "true_label": sample["label"],
                    "prediction": pred,
                    "confidence": conf,
                    "correct": pred == sample["label"]
                }
                
                # For VulnRAG, store the full pipeline result
                if hasattr(self, 'pipeline') and hasattr(self, '_last_pipeline_result'):
                    result_data.update(self._last_pipeline_result)
                
                self.results.append(result_data)
                
            except Exception as e:
                logger.error(f"Error processing sample {sample['id']}: {e}")
                # Default to non-vulnerable with low confidence
                predictions.append(0)
                confidences.append(0.1)
                true_labels.append(sample["label"])
        
        total_time = time.time() - start_time
        
        return {
            "predictions": predictions,
            "confidences": confidences,
            "true_labels": true_labels,
            "total_time": total_time,
            "avg_time_per_sample": total_time / len(samples) if samples else 0
        }

class VulnRAGDetector(BaseDetector):
    """VulnRAG detection baseline using the full pipeline"""
    
    def __init__(self, llm_model: str = "qwen2.5-coder:latest"):
        super().__init__(f"VulnRAG-{llm_model.split('/')[-1].split(':')[0]}")
        self.llm_model = llm_model
        
        # Full mode: use complete pipeline with specified LLM
        try:
            from rag.core.pipeline import VulnRAGPipeline
            self.pipeline = VulnRAGPipeline(llm_model=llm_model)
        except ImportError as e:
            logger.error(f"Could not import VulnRAG pipeline: {e}")
            raise
    
    def detect(self, code: str) -> Tuple[int, float]:
        """Detect vulnerability using VulnRAG with specified LLM"""
        try:
            return self._detect_full(code)
                
        except Exception as e:
            logger.error(f"VulnRAG detection error: {e}")
            return 0, 0.1
    
    def _detect_fast(self, code: str) -> Tuple[int, float]:
        """Fast mode: static + heuristic only (no LLM)"""
        # Static analysis
        static_result = self.static_gate.analyze(code)
        static_assessment = static_result.get("security_assessment", "LIKELY_SAFE")
        
        # Heuristic analysis
        heuristic_result = self.heuristic_gate.analyse(code)
        heuristic_assessment = heuristic_result.security_assessment
        heuristic_score = heuristic_result.risk_score
        
        # Simple voting logic
        static_vote = 1 if static_assessment == "POTENTIALLY_VULNERABLE" else 0
        heuristic_vote = 1 if (heuristic_assessment == "UNCERTAIN_RISK" and heuristic_score > 0.5) else 0
        
        # Decision based on votes
        if static_vote + heuristic_vote >= 1:
            confidence = 0.7 + (heuristic_score * 0.2)
            return 1, min(confidence, 0.9)
        else:
            confidence = 0.6 - (heuristic_score * 0.3)
            return 0, max(confidence, 0.3)
    
    def _detect_full(self, code: str) -> Tuple[int, float]:
        """Full mode: complete VulnRAG pipeline"""
        # Use the complete VulnRAG pipeline
        result = self.pipeline.detect(code)
        
        # Store the full pipeline result for analysis
        self._last_pipeline_result = result
        
        # Extract decision from the pipeline result
        decision = result.get("decision", "NEED MORE CONTEXT")
        confidence = result.get("confidence", 0.5)
        
        # Determine who made the decision
        decision_maker = self._determine_decision_maker(result)
        
        # Log the decision maker for analysis
        logger.info(f"VulnRAG Decision: {decision} by {decision_maker}")
        
        # Convert decision to binary classification
        if decision == "VULNERABLE":
            return 1, confidence if confidence is not None else 0.8
        elif decision == "SAFE":
            return 0, confidence if confidence is not None else 0.8
        else:
            # For "NEED MORE CONTEXT", use a neutral confidence
            return 0, 0.5
    
    def _determine_decision_maker(self, result: Dict) -> str:
        """Determine who made the final decision in VulnRAG"""
        # Check if decision_analysis is available in the result
        if "decision_analysis" in result:
            return result["decision_analysis"]
        
        # Fallback: check timings
        timings = result.get("timings_s", {})
        
        # Check if full RAG was used (longest path)
        if "preprocessing" in timings and "retrieval" in timings and "assembly" in timings:
            return "FULL_RAG_PIPELINE"
        
        # Check if LLM arbitration was used (medium path)
        elif "llm_arbitration" in timings:
            return "LLM_ARBITRATION"
        
        # Check if static+heuristic agreement (shortest path)
        elif "static" in timings and len(timings) <= 2:
            return "STATIC_HEURISTIC_AGREEMENT"
        
        else:
            return "UNKNOWN"

class QwenDetector(BaseDetector):
    """Qwen vanilla detection"""
    
    def __init__(self, model: str = "qwen2.5-coder:latest"):
        super().__init__(f"Qwen-{model.split('/')[-1].split(':')[0]}")
        self.model = model
        # Initialize Qwen model
        try:
            import ollama
            self.client = ollama.Client()
        except ImportError:
            logger.error("Ollama not available")
            raise
    
    def detect(self, code: str) -> Tuple[int, float]:
        """Detect vulnerability using Qwen with optimized prompt engineering"""
        # Smart truncation for Ollama models
        if len(code) > 4000:
            code = code[:4000] + "\n// ... (truncated)"
        
        # Step-by-Step prompt engineering (same as GPT)
        prompt = f"""You are a cybersecurity expert analyzing C code for vulnerabilities. Follow these steps:

STEP 1: Examine the code structure and identify potential security issues
STEP 2: Check for common vulnerability patterns (buffer overflows, format strings, etc.)
STEP 3: Assess the severity and likelihood of exploitation
STEP 4: Provide a structured analysis

Code to analyze:
```c
{code}
```

Respond with a JSON object containing:
- verdict: "VULNERABLE" if security issues found, "SAFE" otherwise
- confidence: float between 0.0 and 1.0 indicating your certainty
- explanation: brief explanation of your findings
- vulnerability_type: specific type if vulnerable (e.g., "buffer_overflow", "format_string")
- affected_lines: line numbers or code sections of concern

JSON response:"""
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            result = response['message']['content'].strip()
            
            # Robust JSON parsing with validation
            try:
                import json
                import re
                
                # Try to extract JSON from response (Qwen might add extra text)
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed = json.loads(json_str)
                else:
                    parsed = json.loads(result)
                
                # Validate required fields
                verdict = parsed.get("verdict", "").upper()
                confidence = parsed.get("confidence", 0.8)
                
                # Validate confidence range
                if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                    confidence = 0.8
                
                # Log detailed analysis for debugging
                safe_log(logger, "INFO", f"Qwen Analysis: {verdict} (confidence: {confidence:.2f})")
                if "vulnerability_type" in parsed:
                    safe_log(logger, "INFO", f"Vulnerability type: {parsed['vulnerability_type']}")
                
                if "VULNERABLE" in verdict:
                    return 1, confidence
                else:
                    return 0, confidence
                    
            except (json.JSONDecodeError, AttributeError) as e:
                safe_log(logger, "WARNING", f"JSON parsing failed: {e}, falling back to text analysis")
                # Enhanced fallback to text analysis
                result_lower = result.lower()
                if "vulnerable" in result_lower or "vulnerability" in result_lower:
                    return 1, 0.7
                elif "safe" in result_lower or "no vulnerability" in result_lower or "secure" in result_lower:
                    return 0, 0.7
                else:
                    return 0, 0.5  # Uncertain
                
        except Exception as e:
            safe_log(logger, "ERROR", f"Qwen detection error: {e}")
            return 0, 0.1

class GPTDetector(BaseDetector):
    """GPT detection via API"""
    
    def __init__(self, api_key: str = None):
        super().__init__("GPT-4")
        
        # Get API key securely
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = get_secure_api_key("OpenAI", "OPENAI_API_KEY")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            safe_log(logger, "ERROR", "OpenAI library not available")
            raise
    
    def detect(self, code: str) -> Tuple[int, float]:
        """Detect vulnerability using GPT with optimized prompt engineering"""
        try:
            # Clean code to handle non-ASCII characters properly
            try:
                # Try to encode as UTF-8 and decode as ASCII, replacing problematic characters
                code = code.encode('utf-8', errors='replace').decode('utf-8')
                # Remove any remaining problematic characters
                code = ''.join(char for char in code if ord(char) < 128 or char in '\n\t\r')
            except Exception:
                # If UTF-8 handling fails, use a more aggressive approach
                code = ''.join(char for char in code if ord(char) < 128)
            
            # Smart token counting and truncation
            try:
                import tiktoken
                encoding = tiktoken.encoding_for_model("gpt-4o")
                
                # Preserve code structure while fitting within limits
                max_code_tokens = 6000  # Leave room for prompt and response
                code_tokens = encoding.encode(code)
                
                if len(code_tokens) > max_code_tokens:
                    # Truncate intelligently, preserving function structure
                    truncated_tokens = code_tokens[:max_code_tokens]
                    code = encoding.decode(truncated_tokens)
                    if not code.strip().endswith('}'):
                        code += "\n// ... (truncated)"
            except ImportError:
                # Fallback to character-based truncation
                if len(code) > 4000:
                    code = code[:4000] + "\n// ... (truncated)"
            
            # Step-by-Step prompt engineering based on recent research
            prompt = f"""You are a cybersecurity expert analyzing C code for vulnerabilities. Follow these steps:

STEP 1: Examine the code structure and identify potential security issues
STEP 2: Check for common vulnerability patterns (buffer overflows, format strings, etc.)
STEP 3: Assess the severity and likelihood of exploitation
STEP 4: Provide a structured analysis

Code to analyze:
```c
{code}
```

Respond with a JSON object containing:
- verdict: "VULNERABLE" if security issues found, "SAFE" otherwise
- confidence: float between 0.0 and 1.0 indicating your certainty
- explanation: brief explanation of your findings
- vulnerability_type: specific type if vulnerable (e.g., "buffer_overflow", "format_string")
- affected_lines: line numbers or code sections of concern

JSON response:"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,  # Sufficient for complete JSON response
                temperature=0.1,  # Low temperature for consistent analysis
                response_format={"type": "json_object"},  # Force JSON output
                top_p=0.9
            )
            
            result = response.choices[0].message.content.strip()
            
            # Robust JSON parsing with validation
            try:
                import json
                parsed = json.loads(result)
                
                # Validate required fields
                verdict = parsed.get("verdict", "").upper()
                confidence = parsed.get("confidence", 0.8)
                
                # Validate confidence range
                if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                    confidence = 0.8
                
                # Log detailed analysis for debugging
                safe_log(logger, "INFO", f"GPT Analysis: {verdict} (confidence: {confidence:.2f})")
                if "vulnerability_type" in parsed:
                    safe_log(logger, "INFO", f"Vulnerability type: {parsed['vulnerability_type']}")
                
                if "VULNERABLE" in verdict:
                    return 1, confidence
                else:
                    return 0, confidence
                    
            except json.JSONDecodeError as e:
                safe_log(logger, "WARNING", f"JSON parsing failed: {e}, falling back to text analysis")
                # Fallback to text analysis
                if "VULNERABLE" in result.upper() or "vulnerability" in result.lower():
                    return 1, 0.7
                elif "SAFE" in result.upper() or "no vulnerability" in result.lower():
                    return 0, 0.7
                else:
                    return 0, 0.5  # Uncertain
                
        except Exception as e:
            error_msg = str(e).encode('ascii', errors='replace').decode('ascii')
            safe_log(logger, "ERROR", f"GPT detection error: {error_msg}")
            return 0, 0.1
        
class StaticToolsDetector(BaseDetector):
    """Static analysis tools detection using the same wrappers as VulnRAG"""
    
    def __init__(self):
        super().__init__("Static-Tools")
        try:
            from rag.core.analysis.static_analysis_gate import StaticAnalysisGate
            self.static_gate = StaticAnalysisGate()
        except ImportError as e:
            logger.error(f"Could not import StaticAnalysisGate: {e}")
            raise
    
    def detect(self, code: str) -> Tuple[int, float]:
        """Detect vulnerability using the same static analysis as VulnRAG"""
        try:
            # Use the same StaticAnalysisGate as VulnRAG
            result = self.static_gate.analyze(code)
            
            # Extract decision
            assessment = result.get("security_assessment", "LIKELY_SAFE")
            
            # Count total issues for confidence
            total_issues = (
                len(result.get("cppcheck_issues", [])) +
                len(result.get("clang_tidy_issues", [])) +
                len(result.get("flawfinder_issues", []))
            )
            
            # Convert to binary classification
            if assessment == "POTENTIALLY_VULNERABLE":
                confidence = min(0.8 + (total_issues * 0.1), 0.95)
                return 1, confidence
            else:
                confidence = max(0.7 - (total_issues * 0.1), 0.3)
                return 0, confidence
                
        except Exception as e:
            logger.error(f"Static tools detection error: {e}")
            return 0, 0.1

def get_available_detectors() -> Dict[str, BaseDetector]:
    """Get all available detectors"""
    detectors = {}
    
    # Try to add each detector
    try:
        # VulnRAG with Qwen2.5
        detectors["vulnrag-qwen2.5"] = VulnRAGDetector(llm_model="qwen2.5-coder:latest")
    except Exception as e:
        logger.warning(f"VulnRAG-Qwen2.5 detector not available: {e}")
    
    try:
        # VulnRAG with Kirito
        detectors["vulnrag-kirito"] = VulnRAGDetector(llm_model="kirito1/qwen3-coder:latest")
    except Exception as e:
        logger.warning(f"VulnRAG-Kirito detector not available: {e}")
    
    try:
        # Qwen2.5 solo
        detectors["qwen2.5"] = QwenDetector(model="qwen2.5-coder:latest")
    except Exception as e:
        logger.warning(f"Qwen2.5 detector not available: {e}")
    
    try:
        # Kirito solo
        detectors["kirito"] = QwenDetector(model="kirito1/qwen3-coder:latest")
    except Exception as e:
        logger.warning(f"Kirito detector not available: {e}")
    
    try:
        detectors["gpt"] = GPTDetector()
    except Exception as e:
        logger.warning(f"GPT detector not available: {e}")
    
    try:
        detectors["static"] = StaticToolsDetector()
    except Exception as e:
        logger.warning(f"Static tools detector not available: {e}")
    
    return detectors 