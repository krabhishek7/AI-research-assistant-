"""Embeddings module for generating embeddings using Hugging Face models."""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

from backend.database.models import Paper
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingsGenerator:
    """Handles generation of embeddings for papers and queries."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = settings.embedding_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 512  # Maximum token length
        self.batch_size = 16   # Batch size for processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Use SentenceTransformer for better scientific paper embeddings
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            logger.info("Falling back to default model...")
            
            # Fallback to a reliable model
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                logger.info("Fallback model loaded successfully")
            except Exception as e2:
                logger.error(f"Error loading fallback model: {str(e2)}")
                raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding generation."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate to max length (roughly)
        words = text.split()
        if len(words) > self.max_length:
            text = ' '.join(words[:self.max_length])
        
        return text
    
    def _create_paper_text(self, paper: Paper) -> str:
        """Create text representation of a paper for embedding."""
        text_parts = []
        
        # Add title (weighted more heavily)
        if paper.title:
            text_parts.append(f"Title: {paper.title}")
        
        # Add authors
        if paper.authors:
            authors_text = ", ".join([author.name for author in paper.authors])
            text_parts.append(f"Authors: {authors_text}")
        
        # Add abstract (most important for semantic search)
        if paper.abstract:
            text_parts.append(f"Abstract: {paper.abstract}")
        
        # Add categories/keywords if available
        if paper.categories:
            text_parts.append(f"Categories: {', '.join(paper.categories)}")
        
        if paper.keywords:
            text_parts.append(f"Keywords: {', '.join(paper.keywords)}")
        
        # Add a portion of full text if available
        if paper.full_text:
            # Use first 1000 characters of full text
            text_parts.append(f"Content: {paper.full_text[:1000]}")
        
        return " ".join(text_parts)
    
    async def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text:
            return None
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Generate embedding using thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                self.model.encode,
                processed_text
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    async def generate_embeddings(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Preprocess all texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Filter out empty texts
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(processed_texts):
                if text:
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if not valid_texts:
                return [None] * len(texts)
            
            # Generate embeddings in batches
            embeddings = []
            for i in range(0, len(valid_texts), self.batch_size):
                batch_texts = valid_texts[i:i + self.batch_size]
                
                # Generate embeddings using thread pool
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    self.executor,
                    self.model.encode,
                    batch_texts
                )
                
                embeddings.extend(batch_embeddings)
            
            # Map embeddings back to original order
            result = [None] * len(texts)
            for i, embedding in enumerate(embeddings):
                result[valid_indices[i]] = embedding
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return [None] * len(texts)
    
    async def generate_paper_embedding(self, paper: Paper) -> Optional[np.ndarray]:
        """
        Generate embedding for a paper.
        
        Args:
            paper: Paper object to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Create text representation of the paper
            paper_text = self._create_paper_text(paper)
            
            # Generate embedding
            embedding = await self.generate_embedding(paper_text)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating paper embedding: {str(e)}")
            return None
    
    async def generate_paper_embeddings(self, papers: List[Paper]) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple papers.
        
        Args:
            papers: List of Paper objects to embed
            
        Returns:
            List of embedding vectors
        """
        if not papers:
            return []
        
        try:
            # Create text representations for all papers
            paper_texts = [self._create_paper_text(paper) for paper in papers]
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(paper_texts)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating paper embeddings: {str(e)}")
            return [None] * len(papers)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            # Ensure similarity is between 0 and 1
            similarity = max(0, min(1, (similarity + 1) / 2))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def find_similar_papers(
        self,
        query_embedding: np.ndarray,
        paper_embeddings: List[np.ndarray],
        papers: List[Paper],
        top_k: int = 10
    ) -> List[Paper]:
        """
        Find papers most similar to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            paper_embeddings: List of paper embedding vectors
            papers: List of corresponding Paper objects
            top_k: Number of top results to return
            
        Returns:
            List of Paper objects sorted by similarity
        """
        try:
            similarities = []
            
            for i, paper_embedding in enumerate(paper_embeddings):
                if paper_embedding is not None:
                    similarity = self.compute_similarity(query_embedding, paper_embedding)
                    similarities.append((similarity, i))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Return top k papers with similarity scores
            result = []
            for similarity, index in similarities[:top_k]:
                paper = papers[index]
                paper.relevance_score = similarity
                result.append(paper)
            
            return result
            
        except Exception as e:
            logger.error(f"Error finding similar papers: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "model_loaded": self.model is not None
        }
    
    def __del__(self):
        """Clean up executor when instance is destroyed."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

# Global instance
embeddings_generator = EmbeddingsGenerator() 