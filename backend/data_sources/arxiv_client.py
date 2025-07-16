"""ArXiv API client for fetching academic papers."""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import quote
import sys
from pathlib import Path

import arxiv
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.database.models import Paper, Author, PaperSource
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivClient:
    """Client for interacting with the ArXiv API."""
    
    def __init__(self):
        self.client = arxiv.Client(
            page_size=settings.arxiv_max_results,
            delay_seconds=settings.arxiv_rate_limit,
            num_retries=3
        )
        self.rate_limit = settings.arxiv_rate_limit
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search_papers(
        self,
        query: str,
        max_results: int = 10,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
        categories: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[Paper]:
        """
        Search for papers on ArXiv.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sort_by: Sort criterion (Relevance, LastUpdatedDate, SubmittedDate)
            sort_order: Sort order (Ascending, Descending)
            categories: List of ArXiv categories to filter by
            date_from: Start date for filtering
            date_to: End date for filtering
            
        Returns:
            List of Paper objects
        """
        try:
            # Build search query
            search_query = self._build_search_query(
                query, categories, date_from, date_to
            )
            
            logger.info(f"Searching ArXiv with query: {search_query}")
            
            # Create search object
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=sort_by,
                sort_order=sort_order
            )
            
            # Execute search
            papers = []
            for result in self.client.results(search):
                paper = self._convert_to_paper(result)
                if paper:
                    papers.append(paper)
                    
            logger.info(f"Found {len(papers)} papers from ArXiv")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {str(e)}")
            raise
    
    def _build_search_query(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> str:
        """Build ArXiv search query with filters."""
        search_parts = []
        
        # Add main query
        if query:
            # Search in title, abstract, and comments
            search_parts.append(f"(ti:{query} OR abs:{query} OR co:{query})")
        
        # Add category filter
        if categories:
            category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            search_parts.append(f"({category_filter})")
        
        # Add date filter (ArXiv uses submitted date)
        if date_from or date_to:
            date_filter = "submittedDate:"
            if date_from and date_to:
                date_filter += f"[{date_from.strftime('%Y%m%d')}* TO {date_to.strftime('%Y%m%d')}*]"
            elif date_from:
                date_filter += f"[{date_from.strftime('%Y%m%d')}* TO *]"
            elif date_to:
                date_filter += f"[* TO {date_to.strftime('%Y%m%d')}*]"
            search_parts.append(date_filter)
        
        return " AND ".join(search_parts) if search_parts else query
    
    def _convert_to_paper(self, arxiv_result: arxiv.Result) -> Optional[Paper]:
        """Convert ArXiv result to Paper object."""
        try:
            # Extract authors
            authors = []
            for author in arxiv_result.authors:
                authors.append(Author(name=str(author)))
            
            # Extract categories
            categories = [str(cat) for cat in arxiv_result.categories]
            
            # Create Paper object
            paper = Paper(
                id=arxiv_result.entry_id,
                title=arxiv_result.title,
                authors=authors,
                abstract=arxiv_result.summary,
                publication_date=arxiv_result.published,
                doi=arxiv_result.doi,
                arxiv_id=arxiv_result.entry_id.split('/')[-1],
                url=arxiv_result.entry_id,
                pdf_url=arxiv_result.pdf_url,
                source=PaperSource.ARXIV,
                categories=categories,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            return paper
            
        except Exception as e:
            logger.error(f"Error converting ArXiv result to Paper: {str(e)}")
            return None
    
    async def get_paper_by_id(self, arxiv_id: str) -> Optional[Paper]:
        """
        Get a specific paper by ArXiv ID.
        
        Args:
            arxiv_id: ArXiv paper ID
            
        Returns:
            Paper object or None if not found
        """
        try:
            # Clean the ID (remove arxiv: prefix if present)
            clean_id = arxiv_id.replace('arxiv:', '')
            
            search = arxiv.Search(id_list=[clean_id])
            
            for result in self.client.results(search):
                return self._convert_to_paper(result)
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting paper by ID {arxiv_id}: {str(e)}")
            return None
    
    async def get_papers_by_author(self, author_name: str, max_results: int = 20) -> List[Paper]:
        """
        Get papers by author name.
        
        Args:
            author_name: Author name to search for
            max_results: Maximum number of results
            
        Returns:
            List of Paper objects
        """
        try:
            search_query = f"au:{author_name}"
            
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in self.client.results(search):
                paper = self._convert_to_paper(result)
                if paper:
                    papers.append(paper)
                    
            return papers
            
        except Exception as e:
            logger.error(f"Error searching papers by author {author_name}: {str(e)}")
            return []
    
    async def get_papers_by_category(self, category: str, max_results: int = 20) -> List[Paper]:
        """
        Get papers by ArXiv category.
        
        Args:
            category: ArXiv category (e.g., 'cs.AI', 'math.CO')
            max_results: Maximum number of results
            
        Returns:
            List of Paper objects
        """
        try:
            search_query = f"cat:{category}"
            
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in self.client.results(search):
                paper = self._convert_to_paper(result)
                if paper:
                    papers.append(paper)
                    
            return papers
            
        except Exception as e:
            logger.error(f"Error searching papers by category {category}: {str(e)}")
            return []
    
    def get_popular_categories(self) -> Dict[str, str]:
        """Get popular ArXiv categories with descriptions."""
        return {
            "cs.AI": "Artificial Intelligence",
            "cs.CL": "Computation and Language",
            "cs.CV": "Computer Vision and Pattern Recognition",
            "cs.LG": "Machine Learning",
            "cs.NE": "Neural and Evolutionary Computing",
            "cs.RO": "Robotics",
            "math.CO": "Combinatorics",
            "math.ST": "Statistics Theory",
            "stat.ML": "Machine Learning (Statistics)",
            "q-bio.QM": "Quantitative Methods",
            "physics.data-an": "Data Analysis, Statistics and Probability",
            "econ.EM": "Econometrics",
            "q-fin.ST": "Statistical Finance"
        }

# Global instance
arxiv_client = ArxivClient() 