"""
CPG Extractor - Minimal
=======================

Extraction CPG + embedding 384D avec Joern
"""

import json
import subprocess
import tempfile
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
        
        # Analyse des appels de fonction
        call_vertices = [v for v in vertices if v.get('label') == 'CALL']
        dangerous_calls = []
        all_calls = []
        
        for call_vertex in call_vertices:
            func_name = extract_property_safe(call_vertex, 'NAME')
            if not func_name:
                code = extract_property_safe(call_vertex, 'CODE', '')
                if '(' in code:
                    func_name = code.split('(')[0].strip().split()[-1]
            
            if func_name:
                all_calls.append(func_name)
                
                # Classification selon taxonomie
                if func_name in self.taxonomy:
                    classification = self.taxonomy[func_name]
                    dangerous_calls.append({
                        'function': func_name,
                        'category': classification['category'],
                        'risk_level': classification['risk_level']
                    })
        
        # Métriques de base
        vertex_count = len(vertices)
        edge_count = len(edges)
        
        # Classification de complexité
        if vertex_count > 200:
            size_class = 'large'
        elif vertex_count > 50:
            size_class = 'medium'
        else:
            size_class = 'small'
        
        # Classification de risque
        high_risk_count = sum(1 for call in dangerous_calls if call['risk_level'] == 'high')
        if high_risk_count > 0:
            risk_class = 'high'
        elif len(dangerous_calls) > 2:
            risk_class = 'medium'
        elif len(dangerous_calls) > 0:
            risk_class = 'low'
        else:
            risk_class = 'unknown'
        
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