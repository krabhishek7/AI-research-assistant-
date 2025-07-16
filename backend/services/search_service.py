"""Search service for semantic and hybrid search implementation."""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import time

from backend.database.models import Paper, PaperSource, SearchQuery, SearchResult
from backend.database.chromadb_client import chromadb_client
from backend.data_sources.arxiv_client import arxiv_client
from backend.data_sources.pubmed_client import pubmed_client
from backend.data_sources.scholar_client import google_scholar_client
from backend.processing.embeddings import embeddings_generator
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchService:
    """Comprehensive search service combining multiple sources and search methods."""
    
    def __init__(self):
        self.chromadb_client = chromadb_client
        self.arxiv_client = arxiv_client
        self.pubmed_client = pubmed_client
        self.google_scholar_client = google_scholar_client
        self.embeddings_generator = embeddings_generator
        self.default_max_results = settings.default_search_results
        self.max_results_limit = settings.max_search_results
    
    async def search_papers(
        self,
        query: str,
        max_results: int = None,
        sources: Optional[List[PaperSource]] = None,
        use_local_db: bool = True,
        use_external_apis: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        categories: Optional[List[str]] = None
    ) -> SearchResult:
        """
        Comprehensive search across local database and external APIs.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sources: List of sources to search (if None, search all available)
            use_local_db: Whether to search local ChromaDB
            use_external_apis: Whether to search external APIs
            filters: Additional metadata filters
            date_from: Start date for filtering
            date_to: End date for filtering
            categories: List of categories to filter by
            
        Returns:
            SearchResult object with combined results
        """
        start_time = time.time()
        
        if max_results is None:
            max_results = self.default_max_results
        
        max_results = min(max_results, self.max_results_limit)
        
        if sources is None:
            sources = [PaperSource.ARXIV, PaperSource.PUBMED, PaperSource.GOOGLE_SCHOLAR]  # Default to all sources
        
        logger.info(f"Starting search for query: '{query}' with max_results: {max_results}")
        
        all_papers = []
        sources_searched = []
        
        # Search local database first
        if use_local_db:
            try:
                local_papers = await self.semantic_search(
                    query=query,
                    max_results=max_results,
                    sources=sources,
                    filters=filters
                )
                all_papers.extend(local_papers)
                logger.info(f"Found {len(local_papers)} papers in local database")
            except Exception as e:
                logger.error(f"Error searching local database: {str(e)}")
        
        # Search external APIs if needed
        if use_external_apis and len(all_papers) < max_results:
            remaining_results = max_results - len(all_papers)
            
            for source in sources:
                if source == PaperSource.ARXIV:
                    try:
                        arxiv_papers = await self.arxiv_client.search_papers(
                            query=query,
                            max_results=remaining_results,
                            categories=categories,
                            date_from=date_from,
                            date_to=date_to
                        )
                        
                        # Filter out papers already in results
                        existing_ids = {p.id for p in all_papers if p.id}
                        new_papers = [p for p in arxiv_papers if p.id not in existing_ids]
                        
                        all_papers.extend(new_papers)
                        sources_searched.append(source)
                        
                        logger.info(f"Found {len(new_papers)} new papers from ArXiv")
                        
                        # Add papers to local database for future searches
                        if new_papers:
                            await self.chromadb_client.add_papers(new_papers)
                        
                    except Exception as e:
                        logger.error(f"Error searching ArXiv: {str(e)}")
                        continue
                
                elif source == PaperSource.PUBMED:
                    try:
                        pubmed_papers = await self.pubmed_client.search_papers(
                            query=query,
                            max_results=remaining_results,
                            date_from=date_from,
                            date_to=date_to
                        )
                        
                        # Filter out papers already in results
                        existing_ids = {p.id for p in all_papers if p.id}
                        new_papers = [p for p in pubmed_papers if p.id not in existing_ids]
                        
                        all_papers.extend(new_papers)
                        sources_searched.append(source)
                        
                        logger.info(f"Found {len(new_papers)} new papers from PubMed")
                        
                        # Add papers to local database for future searches
                        if new_papers:
                            await self.chromadb_client.add_papers(new_papers)
                        
                    except Exception as e:
                        logger.error(f"Error searching PubMed: {str(e)}")
                        continue
                
                elif source == PaperSource.GOOGLE_SCHOLAR:
                    try:
                        logger.info(f"Searching Google Scholar for: {query}")
                        scholar_papers = await self.google_scholar_client.search_papers(
                            query=query,
                            max_results=remaining_results,
                            date_from=date_from,
                            date_to=date_to
                        )
                        
                        logger.info(f"Google Scholar client returned {len(scholar_papers)} papers")
                        
                        # Filter out papers already in results
                        existing_ids = {p.id for p in all_papers if p.id}
                        logger.info(f"Existing paper IDs: {existing_ids}")
                        logger.info(f"Google Scholar paper IDs: {[p.id for p in scholar_papers]}")
                        new_papers = [p for p in scholar_papers if p.id not in existing_ids]
                        
                        logger.info(f"After filtering, {len(new_papers)} new papers from Google Scholar")
                        
                        all_papers.extend(new_papers)
                        sources_searched.append(source)
                        
                        logger.info(f"Total papers after Google Scholar: {len(all_papers)}")
                        
                        # Add papers to local database for future searches
                        if new_papers:
                            try:
                                await self.chromadb_client.add_papers(new_papers)
                                logger.info(f"Added {len(new_papers)} Google Scholar papers to database")
                            except Exception as db_error:
                                logger.error(f"Error adding Google Scholar papers to database: {str(db_error)}")
                        
                    except Exception as e:
                        logger.error(f"Error searching Google Scholar: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # Even if Google Scholar fails, mark it as searched
                        sources_searched.append(source)
                        continue
        
        # Re-rank results using semantic similarity if we have mixed sources
        if len(all_papers) > 1:
            # TODO: Implement reranking based on semantic similarity
            logger.info(f"Skipping reranking for {len(all_papers)} papers")
        
        # Limit to max_results
        all_papers = all_papers[:max_results]
        
        search_time = time.time() - start_time
        
        return SearchResult(
            papers=all_papers,
            total_results=len(all_papers),
            query=query,
            search_time=search_time,
            sources_searched=sources_searched
        )
    
    async def semantic_search(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[List[PaperSource]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Paper]:
        """
        Semantic search using embeddings in local ChromaDB.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sources: List of sources to search
            filters: Additional metadata filters
            
        Returns:
            List of Paper objects ordered by relevance
        """
        try:
            # Search ChromaDB using semantic similarity
            papers = await self.chromadb_client.search_papers(
                query=query,
                max_results=max_results,
                sources=sources,
                filters=filters
            )
            
            return papers
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    async def hybrid_search(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[List[PaperSource]] = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Paper]:
        """
        Hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sources: List of sources to search
            semantic_weight: Weight for semantic search results
            keyword_weight: Weight for keyword search results
            
        Returns:
            List of Paper objects ordered by combined relevance
        """
        try:
            # Perform semantic search
            semantic_results = await self.semantic_search(
                query=query,
                max_results=max_results * 2,  # Get more results for better hybrid ranking
                sources=sources
            )
            
            # Perform keyword search (simple implementation)
            keyword_results = await self._keyword_search(
                query=query,
                max_results=max_results * 2,
                sources=sources
            )
            
            # Combine and re-rank results
            combined_results = await self._combine_search_results(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )
            
            return combined_results[:max_results]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[List[PaperSource]] = None
    ) -> List[Paper]:
        """
        Simple keyword search implementation.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sources: List of sources to search
            
        Returns:
            List of Paper objects ordered by keyword relevance
        """
        # This is a simplified implementation
        # In a full system, you'd use a proper text search engine like Elasticsearch
        
        query_terms = query.lower().split()
        
        # Search in ChromaDB with metadata filtering
        # For now, we'll use the semantic search and manually score keyword matches
        papers = await self.semantic_search(
            query=query,
            max_results=max_results * 3,  # Get more results for keyword filtering
            sources=sources
        )
        
        # Score papers based on keyword matches
        scored_papers = []
        for paper in papers:
            score = self._compute_keyword_score(paper, query_terms)
            if score > 0:
                paper.relevance_score = score
                scored_papers.append(paper)
        
        # Sort by keyword score
        scored_papers.sort(key=lambda p: p.relevance_score or 0, reverse=True)
        
        return scored_papers[:max_results]
    
    def _compute_keyword_score(self, paper: Paper, query_terms: List[str]) -> float:
        """Compute keyword matching score for a paper."""
        score = 0.0
        
        # Check title (highest weight)
        title_lower = paper.title.lower()
        for term in query_terms:
            if term in title_lower:
                score += 2.0
        
        # Check abstract (medium weight)
        abstract_lower = paper.abstract.lower()
        for term in query_terms:
            if term in abstract_lower:
                score += 1.0
        
        # Check categories and keywords (lower weight)
        all_categories = ' '.join(paper.categories).lower() if paper.categories else ''
        all_keywords = ' '.join(paper.keywords).lower() if paper.keywords else ''
        
        for term in query_terms:
            if term in all_categories:
                score += 0.5
            if term in all_keywords:
                score += 0.5
        
        return score
    
    async def _combine_search_results(
        self,
        semantic_results: List[Paper],
        keyword_results: List[Paper],
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Paper]:
        """Combine semantic and keyword search results."""
        # Create a dictionary to store combined scores
        paper_scores = {}
        
        # Add semantic results
        for paper in semantic_results:
            paper_id = paper.id or paper.title
            semantic_score = paper.relevance_score or 0
            paper_scores[paper_id] = {
                'paper': paper,
                'semantic_score': semantic_score,
                'keyword_score': 0.0
            }
        
        # Add keyword results
        for paper in keyword_results:
            paper_id = paper.id or paper.title
            keyword_score = paper.relevance_score or 0
            
            if paper_id in paper_scores:
                paper_scores[paper_id]['keyword_score'] = keyword_score
            else:
                paper_scores[paper_id] = {
                    'paper': paper,
                    'semantic_score': 0.0,
                    'keyword_score': keyword_score
                }
        
        # Compute combined scores
        combined_results = []
        for paper_id, scores in paper_scores.items():
            combined_score = (
                scores['semantic_score'] * semantic_weight +
                scores['keyword_score'] * keyword_weight
            )
            
            paper = scores['paper']
            paper.relevance_score = combined_score
            combined_results.append(paper)
        
        # Sort by combined score
        combined_results.sort(key=lambda p: p.relevance_score or 0, reverse=True)
        
        return combined_results
    
    async def _rerank_papers(self, query: str, papers: List[Paper]) -> List[Paper]:
        """Re-rank papers using semantic similarity."""
        try:
            # Generate query embedding
            query_embedding = await self.embeddings_generator.generate_embedding(query)
            
            if query_embedding is None:
                return papers
            
            # Generate embeddings for papers that don't have relevance scores
            papers_to_embed = [p for p in papers if p.relevance_score is None]
            
            if papers_to_embed:
                paper_embeddings = await self.embeddings_generator.generate_paper_embeddings(
                    papers_to_embed
                )
                
                # Compute similarity scores
                for i, paper in enumerate(papers_to_embed):
                    if paper_embeddings[i] is not None:
                        similarity = self.embeddings_generator.compute_similarity(
                            query_embedding, paper_embeddings[i]
                        )
                        paper.relevance_score = similarity
            
            # Sort all papers by relevance score
            papers.sort(key=lambda p: p.relevance_score or 0, reverse=True)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error re-ranking papers: {str(e)}")
            return papers
    
    async def find_related_papers(
        self,
        paper: Paper,
        max_results: int = 10,
        sources: Optional[List[PaperSource]] = None
    ) -> List[Paper]:
        """
        Find papers related to a given paper.
        
        Args:
            paper: Reference paper
            max_results: Maximum number of results to return
            sources: List of sources to search
            
        Returns:
            List of related Paper objects
        """
        try:
            # Create query from paper content
            query_parts = []
            
            if paper.title:
                query_parts.append(paper.title)
            
            if paper.abstract:
                # Use first 500 characters of abstract
                query_parts.append(paper.abstract[:500])
            
            if paper.keywords:
                query_parts.extend(paper.keywords[:5])  # Top 5 keywords
            
            query = ' '.join(query_parts)
            
            # Search for related papers
            related_papers = await self.semantic_search(
                query=query,
                max_results=max_results + 1,  # +1 to account for the original paper
                sources=sources
            )
            
            # Remove the original paper from results
            related_papers = [p for p in related_papers if p.id != paper.id]
            
            return related_papers[:max_results]
            
        except Exception as e:
            logger.error(f"Error finding related papers: {str(e)}")
            return []
    
    async def get_search_suggestions(self, partial_query: str) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            
        Returns:
            List of suggested queries
        """
        try:
            # This is a simple implementation
            # In a full system, you'd use a proper suggestion engine
            
            suggestions = []
            
            # Add some common academic search terms
            common_terms = [
                "machine learning", "deep learning", "neural networks",
                "artificial intelligence", "computer vision", "natural language processing",
                "reinforcement learning", "supervised learning", "unsupervised learning",
                "data mining", "big data", "statistical analysis"
            ]
            
            partial_lower = partial_query.lower()
            
            for term in common_terms:
                if partial_lower in term or term.startswith(partial_lower):
                    suggestions.append(term)
            
            return suggestions[:10]  # Return top 10 suggestions
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {str(e)}")
            return []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search service statistics."""
        return {
            "chromadb_stats": self.chromadb_client.get_collection_stats(),
            "embeddings_model": self.embeddings_generator.get_model_info(),
            "default_max_results": self.default_max_results,
            "max_results_limit": self.max_results_limit
        }

# Global instance
search_service = SearchService() 