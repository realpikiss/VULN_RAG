"""
VulRAG Preprocessing - Minimal Interface
========================================

Interface simple : pipeline + result + factory
"""

from .preprocessing import PreprocessingPipeline, PreprocessingResult, create_pipeline

__all__ = ['PreprocessingPipeline', 'PreprocessingResult', 'create_pipeline']