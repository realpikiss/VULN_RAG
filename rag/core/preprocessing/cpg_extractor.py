"""
CPG Extractor - Minimal
=======================

Extraction CPG + embedding 384D avec Joern
"""

import json
import subprocess
import tempfile
import json
import shutil
from pathlib import Path
from typing import Dict
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

class CPGExtractor:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        
        # Taxonomie complète basée sur dangerous_functions.yaml
        self.taxonomy = {
            # Critical buffer overflow functions
            'strcpy': {'category': 'buffer_overflow', 'risk_level': 'high'},
            'strcat': {'category': 'buffer_overflow', 'risk_level': 'high'},
            'sprintf': {'category': 'buffer_overflow', 'risk_level': 'high'},
            'gets': {'category': 'buffer_overflow', 'risk_level': 'high'},
            'scanf': {'category': 'buffer_overflow', 'risk_level': 'high'},
            'vsprintf': {'category': 'buffer_overflow', 'risk_level': 'high'},
            
            # Memory management risks
            'malloc': {'category': 'memory_management', 'risk_level': 'medium'},
            'free': {'category': 'memory_management', 'risk_level': 'medium'},
            'calloc': {'category': 'memory_management', 'risk_level': 'medium'},
            'realloc': {'category': 'memory_management', 'risk_level': 'medium'},
            'alloca': {'category': 'memory_management', 'risk_level': 'high'},
            'memcpy': {'category': 'memory_management', 'risk_level': 'medium'},
            
            # Format string vulnerabilities
            'printf': {'category': 'format_string', 'risk_level': 'medium'},
            'fprintf': {'category': 'format_string', 'risk_level': 'medium'},
            'syslog': {'category': 'format_string', 'risk_level': 'medium'},
            
            # Command injection
            'system': {'category': 'process_control', 'risk_level': 'high'},
            'exec': {'category': 'process_control', 'risk_level': 'high'},
            'popen': {'category': 'process_control', 'risk_level': 'high'},
        }
        
        # Verify Joern
        subprocess.run(["joern-parse", "--help"], check=True, capture_output=True)
        subprocess.run(["joern-export", "--help"], check=True, capture_output=True)
    
    def extract_embedding(self, code: str) -> np.ndarray:
        cpg_path = self._generate_cpg(code)
        features = self._extract_features(cpg_path)
        text = self._create_text(features)
        
        # Generate 384D embedding
        embedding = self.model.encode([text])[0]
        
        # Normalize and validate
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Ensure 384D float32
        embedding = embedding.astype(np.float32)
        assert embedding.shape == (384,), f"Expected 384D, got {embedding.shape}"
        
        return embedding
    
    def _generate_cpg(self, code: str) -> Path:
        tmpdir = Path(tempfile.mkdtemp())
        code_file = tmpdir / "code.c"
        cpg_file = tmpdir / "cpg.bin"
        export_dir = tmpdir / "export"
    
        # Write code to file
        code_file.write_text(code)
    
        # 1. Parse code to CPG
        parse_cmd = [
            "joern-parse",
            str(code_file),
            "-o", str(cpg_file)
        ]
    
        result = subprocess.run(parse_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"joern-parse failed: {result.stderr}")
    
        # 2. Make sure export directory doesn't exist
        if export_dir.exists():
            shutil.rmtree(export_dir)
    
        # 3. Export CPG to GraphSON
        export_cmd = [
            "joern-export",
            str(cpg_file),
            "--repr=cpg",
            "--format=graphson",
            f"--out={export_dir}"  # joern-export will create this directory
        ]
    
        result = subprocess.run(export_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"joern-export failed: {result.stderr}")
    
        # 4. Look for exported files
        json_files = list(export_dir.rglob("*.json")) + list(export_dir.rglob("*.graphson"))
        if not json_files:
            raise RuntimeError(f"No JSON/GraphSON files found in {export_dir}")
    
        return json_files[0]
    
    def _extract_features(self, cpg_path: Path) -> Dict:
        with open(cpg_path) as f:
            data = json.load(f)
        
        vertices = data['@value']['vertices']
        edges = data['@value']['edges']
        
        # Extraction avec structure GraphSON confirmée
        def extract_property_safe(vertex: Dict, property_name: str, default=None):
            try:
                return vertex['properties'][property_name]['@value']['@value'][0]
            except (KeyError, IndexError, TypeError):
                return default
        
        # Analyze function calls from GraphSON data
        function_calls = []
        for vertex in vertices:
            if extract_property_safe(vertex, 'label') == 'CALL':
                function_calls.append({
                    "name": extract_property_safe(vertex, 'name', 'unknown'),
                    "line": extract_property_safe(vertex, 'lineNumber', 0),
                    "column": extract_property_safe(vertex, 'columnNumber', 0)
                })
        
        # Extract control flow from edges
        control_flow = []
        for edge in edges:
            if extract_property_safe(edge, 'label') == 'CFG':
                control_flow.append({
                    "from": extract_property_safe(edge, 'outV', 'unknown'),
                    "to": extract_property_safe(edge, 'inV', 'unknown'),
                    "type": "control_flow"
                })
        
        # Complexity classification
        complexity_score = len(function_calls) + len(control_flow) * 0.5
        if complexity_score < 10:
            complexity_class = "simple"
        elif complexity_score < 30:
            complexity_class = "medium"
        else:
            complexity_class = "complex"
        
        # Calculate basic metrics
        vertex_count = len(vertices)
        edge_count = len(edges)
        
        # Analyze dangerous function calls
        all_calls = function_calls
        dangerous_calls = []
        for call in function_calls:
            func_name = call['name']
            if func_name in self.taxonomy:
                dangerous_calls.append({
                    'function': func_name,
                    'risk_level': self.taxonomy[func_name]['risk_level'],
                    'category': self.taxonomy[func_name]['category'],
                    'line': call['line']
                })
        
        # Complexity classification
        complexity_score = len(function_calls) + len(control_flow) * 0.5
        if complexity_score < 10:
            size_class = "simple"
            risk_class = "low"
        elif complexity_score < 30:
            size_class = "medium"
            risk_class = "medium" if dangerous_calls else "low"
        else:
            size_class = "complex"
            risk_class = "high" if dangerous_calls else "medium"
        
        return {
            'basic_metrics': {
                'total_vertices': vertex_count,
                'total_edges': edge_count,
                'graph_density': edge_count / vertex_count if vertex_count > 0 else 0
            },
            'call_analysis': {
                'all_calls': all_calls,
                'dangerous_calls': dangerous_calls,
                'dangerous_functions': [call['function'] for call in dangerous_calls]
            },
            'complexity': {
                'size_class': size_class,
                'risk_class': risk_class
            }
        }
    
    def _create_text(self, features: Dict) -> str:
        """Créer texte d'embedding selon format KB2 documenté"""
        text_parts = []
        
        # Informations de base
        basic = features['basic_metrics']
        complexity = features['complexity']
        call_analysis = features['call_analysis']
        
        # Format: "code analysis: X vertices Y edges"
        text_parts.append(f"code analysis: {basic['total_vertices']} vertices {basic['total_edges']} edges")
        
        # Format: "complexity: size_class risk: risk_class"
        text_parts.append(f"complexity: {complexity['size_class']} risk: {complexity['risk_class']}")
        
        # Fonctions dangereuses si présentes
        dangerous_funcs = call_analysis['dangerous_functions']
        if dangerous_funcs:
            text_parts.append(f"dangerous functions: {' '.join(dangerous_funcs[:5])}")
        
        # Distribution des risques
        high_risk = [call for call in call_analysis['dangerous_calls'] if call['risk_level'] == 'high']
        medium_risk = [call for call in call_analysis['dangerous_calls'] if call['risk_level'] == 'medium']
        
        if high_risk or medium_risk:
            risk_desc = []
            if high_risk:
                risk_desc.append(f"high:{len(high_risk)}")
            if medium_risk:
                risk_desc.append(f"medium:{len(medium_risk)}")
            text_parts.append(f"risk distribution: {' '.join(risk_desc)}")
        
        return " | ".join(text_parts)