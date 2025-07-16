"""Processing package for Academic Research Assistant."""

from .paper_processor import PaperProcessor, paper_processor
from .embeddings import EmbeddingsGenerator, embeddings_generator

__all__ = [
    "PaperProcessor", 
    "paper_processor",
    "EmbeddingsGenerator",
    "embeddings_generator"
] 