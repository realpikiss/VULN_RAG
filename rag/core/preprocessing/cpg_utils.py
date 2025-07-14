"""Utility functions for working with Joern CPG exports.

These functions are intended to be the *single source of truth* for:
1. Extracting KB-2 style structural features from a CPG JSON / GraphSON file.
2. Computing a vector embedding from the graph structure.

The real logic already exists in the project notebooks:
    notebooks/03_Kb2_Building.ipynb
    notebooks/04_Kb2_EmbedingAdding_system.ipynb

➡  Copy / port the contents of those notebook cells into the
    corresponding functions below.  Until that is done the current
    implementations serve as safe fall-backs so that the rest of the
    codebase runs and unit-tests pass on machines that do **not** have
    Joern or the embedding model installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "extract_kb2_features",
    "compute_structural_graph_embedding",
]


from collections import Counter

# ---------------------------------------------------------------------------
# Simplified port from notebook 03_Kb2_Building
# ---------------------------------------------------------------------------

def _extract_vertex_info(vertex: Dict) -> Dict:
    """Extract vertex label and flat properties from GraphSON vertex."""
    info = {
        "label": vertex.get("label", ""),
        "properties": {},
    }
    props = vertex.get("properties", {})
    for prop_name, prop_data in props.items():
        # Navigate Joern GraphSON nested @value wrappers
        if isinstance(prop_data, dict) and "@value" in prop_data:
            value_data = prop_data["@value"]
            if isinstance(value_data, dict) and "@value" in value_data:
                values = value_data["@value"]
                if isinstance(values, list) and values:
                    info["properties"][prop_name] = values[0]
                else:
                    info["properties"][prop_name] = values
    return info


def _extract_edge_info(edge: Dict) -> Dict:
    return {
        "label": edge.get("label", ""),
        "from_type": edge.get("outVLabel", ""),
        "to_type": edge.get("inVLabel", ""),
    }


def _analyze_full_cpg(cpg_json_path: Path) -> Dict:
    """Lightweight re-implementation of notebook `analyze_full_cpg`."""
    try:
        with open(cpg_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vertices = data["@value"]["vertices"]
        edges = data["@value"]["edges"]
    except Exception as e:
        logger.warning("Failed to parse GraphSON %s: %s", cpg_json_path, e)
        return {
            "file_info": {"vertex_count": 0, "edge_count": 0},
            "vertex_types": {},
            "dangerous_calls": {},
            "all_calls": {},
            "control_structures": {},
            "identifiers": {},
        }

    analysis = {
        "file_info": {
            "vertex_count": len(vertices),
            "edge_count": len(edges),
        },
        "vertex_types": Counter(),
        "dangerous_calls": Counter(),
        "all_calls": Counter(),
        "control_structures": Counter(),
        "identifiers": Counter(),
    }

    dangerous_functions = {
        "strcpy", "strcat", "sprintf", "scanf", "gets", "strncpy",
        "malloc", "free", "calloc", "realloc", "memcpy", "memmove",
        "memset", "alloca", "delete", "new",
    }

    for vertex in vertices:
        info = _extract_vertex_info(vertex)
        label = info["label"]
        analysis["vertex_types"][label] += 1

        if label == "CALL" and "NAME" in info["properties"]:
            name = info["properties"]["NAME"]
            analysis["all_calls"][name] += 1
            lower = str(name).lower()
            if any(d in lower for d in dangerous_functions):
                analysis["dangerous_calls"][name] += 1
        elif label == "IDENTIFIER" and "NAME" in info["properties"]:
            analysis["identifiers"][info["properties"]["NAME"]] += 1
        elif label in {"CONTROL_STRUCTURE", "IF", "FOR", "WHILE", "BLOCK"}:
            analysis["control_structures"][label] += 1

    # Trim counters to dicts (top-N like notebook)
    def _top(counter: Counter, n: int):
        return dict(counter.most_common(n))

    return {
        "file_info": analysis["file_info"],
        "vertex_types": dict(analysis["vertex_types"]),
        "dangerous_calls": _top(analysis["dangerous_calls"], 10),
        "all_calls": _top(analysis["all_calls"], 20),
        "control_structures": dict(analysis["control_structures"]),
        "identifiers": _top(analysis["identifiers"], 15),
    }


def extract_kb2_features(cpg_json_path: Path) -> Dict:
    """Extract KB-2 style features from a Joern GraphSON/JSON file."""
    analysis = _analyze_full_cpg(cpg_json_path)

    vertex_cnt = analysis["file_info"]["vertex_count"] or 1
    edge_cnt = analysis["file_info"]["edge_count"]

    features = {
        "file_info": {
            "source_file": cpg_json_path.name,
            "size_kb": round(cpg_json_path.stat().st_size / 1024, 1),
            "vertex_count": vertex_cnt,
            "edge_count": edge_cnt,
        },
        "security_features": {
            "dangerous_calls": list(analysis["dangerous_calls"].keys()),
            "dangerous_call_count": sum(analysis["dangerous_calls"].values()),
            "has_malloc_family": any(
                any(x in call for x in ["malloc", "calloc", "realloc"]) for call in analysis["all_calls"].keys()
            ),
            "has_string_functions": any(
                any(x in call.lower() for x in ["strcpy", "strcat", "sprintf", "scanf"]) for call in analysis["all_calls"].keys()
            ),
            "has_memory_functions": any(
                any(x in call.lower() for x in ["memset", "memcpy", "memmove", "free"]) for call in analysis["all_calls"].keys()
            ),
        },
        "code_patterns": {
            "all_calls": dict(list(analysis["all_calls"].items())[:20]),
            "call_count": len(analysis["all_calls"]),
            "control_structure_count": sum(analysis["control_structures"].values()),
            "identifier_count": len(analysis["identifiers"]),
            "vertex_type_distribution": analysis["vertex_types"],
        },
        "complexity_metrics": {
            "call_to_vertex_ratio": len(analysis["all_calls"]) / vertex_cnt,
            "edge_density": edge_cnt / vertex_cnt,
            "control_flow_complexity": analysis["vertex_types"].get("CONTROL_STRUCTURE", 0)
            + analysis["vertex_types"].get("BLOCK", 0),
        },
        "signatures": {
            "dangerous_call_signature": sorted(analysis["dangerous_calls"].keys()),
            "top_calls_signature": sorted(list(analysis["all_calls"].keys())[:10]),
            "vertex_type_signature": sorted(
                [(k, v) for k, v in analysis["vertex_types"].items() if v > 2]
            ),
        },
    }

    return features


# ---------------------------------------------------------------------------
# Simplified port from notebook 04_Kb2_EmbedingAdding_system
# ---------------------------------------------------------------------------

try:
    import networkx as nx  # type: ignore
    _NX_AVAILABLE = True
except Exception:  # pragma: no cover
    _NX_AVAILABLE = False


def _graphson_to_networkx(graphson_data: Dict):
    """Convert Joern GraphSON to undirected NetworkX graph (labels kept)."""
    if not _NX_AVAILABLE:
        return None
    vertices = graphson_data["@value"].get("vertices", [])
    edges = graphson_data["@value"].get("edges", [])
    G = nx.Graph()
    for v in vertices:
        vid = v["id"].get("@value") if isinstance(v.get("id"), dict) else v.get("id")
        G.add_node(vid, label=v.get("label", "UNKNOWN"))
    for e in edges:
        u = e["outV"].get("@value") if isinstance(e.get("outV"), dict) else e.get("outV")
        v_ = e["inV"].get("@value") if isinstance(e.get("inV"), dict) else e.get("inV")
        G.add_edge(u, v_)
    return G


def compute_structural_graph_embedding(cpg_json_path: Path, dim: int = 128) -> np.ndarray:
    """Compute structural embedding via hand-crafted feature vector (no ML deps).

    Falls back to pseudo-random vector when NetworkX is unavailable.
    """
    if not cpg_json_path.exists():
        return np.zeros(dim, dtype=np.float32)

    if not _NX_AVAILABLE:
        # fallback deterministic random
        with open(cpg_json_path, "rb") as f:
            raw = f.read()
        rng = np.random.default_rng(hash(raw) % 2 ** 32)
        vec = rng.random(dim, dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-6
        return vec

    try:
        with open(cpg_json_path, "r", encoding="utf-8") as f:
            graphson = json.load(f)
        G = _graphson_to_networkx(graphson)
        if G is None or G.number_of_nodes() == 0:
            raise ValueError("empty graph or conversion failed")

        features: List[float] = []
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        edge_node_ratio = n_edges / max(n_nodes, 1)
        density = nx.density(G)
        component_ratio = len(list(nx.connected_components(G))) / max(n_nodes, 1)
        features.extend([edge_node_ratio, density, component_ratio])

        label_counts = Counter([G.nodes[n].get("label", "UNKNOWN") for n in G.nodes])
        total_nodes = max(sum(label_counts.values()), 1)
        vuln_labels = [
            "CALL",
            "IDENTIFIER",
            "CONTROL_STRUCTURE",
            "BLOCK",
            "LOCAL",
            "METHOD_PARAMETER_IN",
            "METHOD_PARAMETER_OUT",
            "LITERAL",
            "RETURN",
            "METHOD",
        ]
        for lab in vuln_labels:
            features.append(label_counts.get(lab, 0) / total_nodes)

        degrees = [d for _, d in G.degree()]
        if degrees:
            deg_mean = float(np.mean(degrees))
            deg_std = float(np.std(degrees))
            deg_range = float(np.max(degrees) - np.min(degrees))
            features.extend([
                deg_mean / max(n_nodes, 1),
                deg_std / max(deg_mean, 1) if deg_mean else 0.0,
                deg_range / max(deg_mean, 1) if deg_mean else 0.0,
            ])
            hist, _ = np.histogram(degrees, bins=5, density=True)
            features.extend(hist.tolist())
            hub_thr = np.percentile(degrees, 80)
            hub_prop = sum(1 for d in degrees if d >= hub_thr) / len(degrees)
            features.append(float(hub_prop))
        else:
            features.extend([0.0] * 10)

        patterns = Counter()
        for u, v in G.edges():
            pattern = tuple(sorted([G.nodes[u].get("label", "UNKNOWN"), G.nodes[v].get("label", "UNKNOWN")]))
            patterns[pattern] += 1
        total_edges = max(n_edges, 1)
        crit_patterns = [
            ("CALL", "IDENTIFIER"),
            ("CALL", "CONTROL_STRUCTURE"),
            ("CALL", "LITERAL"),
            ("IDENTIFIER", "IDENTIFIER"),
            ("BLOCK", "CALL"),
            ("CONTROL_STRUCTURE", "BLOCK"),
        ]
        for pat in crit_patterns:
            features.append(patterns.get(pat, 0) / total_edges)

        for lab in vuln_labels[:5]:
            nodes_lab = [n for n in G.nodes if G.nodes[n].get("label") == lab]
            if len(nodes_lab) > 1:
                subg = G.subgraph(nodes_lab)
                clustering = nx.average_clustering(subg) if subg.number_of_edges() > 0 else 0.0
            else:
                clustering = 0.0
            features.append(clustering)

        vec = np.array(features, dtype=np.float32)
        if np.linalg.norm(vec) > 0:
            vec = vec / np.linalg.norm(vec)
        if len(vec) < dim:
            vec = np.concatenate([vec, np.zeros(dim - len(vec), dtype=np.float32)])
        else:
            vec = vec[:dim]
        return vec.astype(np.float32)
    except Exception as e:
        logger.warning("Failed structured embedding; fallback random. Err: %s", e)
        with open(cpg_json_path, "rb") as f:
            raw = f.read()
        rng = np.random.default_rng(hash(raw) % 2 ** 32)
        vec = rng.random(dim, dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-6
        return vec
