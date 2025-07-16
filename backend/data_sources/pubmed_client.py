"""PubMed API client for fetching biomedical literature."""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import aiohttp
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

class PubMedClient:
    """Client for interacting with the PubMed API."""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.search_url = f"{self.base_url}/esearch.fcgi"
        self.fetch_url = f"{self.base_url}/efetch.fcgi"
        self.summary_url = f"{self.base_url}/esummary.fcgi"
        self.api_key = getattr(settings, 'pubmed_api_key', None)
        self.rate_limit = 3  # requests per second (higher with API key)
        self.session = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search_papers(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance",
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        article_types: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Search for papers on PubMed.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sort: Sort order (relevance, date, author)
            date_from: Start date for filtering
            date_to: End date for filtering
            article_types: List of article types to filter by
            
        Returns:
            List of Paper objects
        """
        try:
            # Build search query
            search_query = self._build_search_query(
                query, date_from, date_to, article_types
            )
            
            logger.info(f"Searching PubMed with query: {search_query}")
            
            # First, search for PMIDs
            pmids = await self._search_pmids(search_query, max_results, sort)
            
            if not pmids:
                logger.info("No papers found on PubMed")
                return []
            
            # Fetch paper details
            papers = await self._fetch_papers(pmids)
            
            logger.info(f"Found {len(papers)} papers from PubMed")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            raise
    
    async def _search_pmids(self, query: str, max_results: int, sort: str) -> List[str]:
        """Search for PMIDs using the query."""
        session = await self._get_session()
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "xml",
            "sort": sort
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            async with session.get(self.search_url, params=params) as response:
                response.raise_for_status()
                content = await response.text()
                
                # Parse XML response
                root = ET.fromstring(content)
                pmids = []
                
                for id_elem in root.findall(".//Id"):
                    pmids.append(id_elem.text)
                
                return pmids
                
        except Exception as e:
            logger.error(f"Error searching PMIDs: {str(e)}")
            return []
    
    async def _fetch_papers(self, pmids: List[str]) -> List[Paper]:
        """Fetch paper details for the given PMIDs."""
        if not pmids:
            return []
        
        session = await self._get_session()
        papers = []
        
        # Process PMIDs in batches to avoid overwhelming the API
        batch_size = 20
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            
            params = {
                "db": "pubmed",
                "id": ",".join(batch_pmids),
                "retmode": "xml",
                "rettype": "abstract"
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            try:
                async with session.get(self.fetch_url, params=params) as response:
                    response.raise_for_status()
                    content = await response.text()
                    
                    # Parse XML response
                    batch_papers = self._parse_pubmed_xml(content)
                    papers.extend(batch_papers)
                    
                    # Rate limiting
                    await asyncio.sleep(1.0 / self.rate_limit)
                    
            except Exception as e:
                logger.error(f"Error fetching papers for batch: {str(e)}")
                continue
        
        return papers
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[Paper]:
        """Parse PubMed XML response into Paper objects."""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall(".//PubmedArticle"):
                paper = self._parse_single_article(article)
                if paper:
                    papers.append(paper)
                    
        except Exception as e:
            logger.error(f"Error parsing PubMed XML: {str(e)}")
        
        return papers
    
    def _parse_single_article(self, article_elem) -> Optional[Paper]:
        """Parse a single PubMed article XML element."""
        try:
            # Extract PMID
            pmid_elem = article_elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            
            # Extract title
            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "Unknown Title"
            
            # Extract abstract
            abstract_parts = []
            for abstract_elem in article_elem.findall(".//AbstractText"):
                label = abstract_elem.get("Label", "")
                text = abstract_elem.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            
            abstract = " ".join(abstract_parts) if abstract_parts else None
            
            # Extract authors
            authors = []
            for author_elem in article_elem.findall(".//Author"):
                last_name = author_elem.findtext("LastName", "")
                first_name = author_elem.findtext("ForeName", "")
                initials = author_elem.findtext("Initials", "")
                
                if last_name:
                    authors.append(Author(
                        first_name=first_name,
                        last_name=last_name,
                        initials=initials
                    ))
            
            # Extract publication date
            pub_date = None
            pub_date_elem = article_elem.find(".//PubDate")
            if pub_date_elem is not None:
                year_elem = pub_date_elem.find("Year")
                month_elem = pub_date_elem.find("Month")
                day_elem = pub_date_elem.find("Day")
                
                if year_elem is not None:
                    try:
                        year = int(year_elem.text)
                        month = int(month_elem.text) if month_elem is not None and month_elem.text.isdigit() else 1
                        day = int(day_elem.text) if day_elem is not None and day_elem.text.isdigit() else 1
                        pub_date = datetime(year, month, day)
                    except (ValueError, TypeError):
                        pass
            
            # Extract journal
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else None
            
            # Extract DOI
            doi = None
            for article_id in article_elem.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break
            
            # Extract volume, issue, pages
            volume_elem = article_elem.find(".//Volume")
            volume = volume_elem.text if volume_elem is not None else None
            
            issue_elem = article_elem.find(".//Issue")
            issue = issue_elem.text if issue_elem is not None else None
            
            pages_elem = article_elem.find(".//MedlinePgn")
            pages = pages_elem.text if pages_elem is not None else None
            
            # Extract keywords
            keywords = []
            for keyword_elem in article_elem.findall(".//Keyword"):
                if keyword_elem.text:
                    keywords.append(keyword_elem.text)
            
            # Extract mesh terms
            mesh_terms = []
            for mesh_elem in article_elem.findall(".//MeshHeading/DescriptorName"):
                if mesh_elem.text:
                    mesh_terms.append(mesh_elem.text)
            
            # Combine keywords and mesh terms
            all_keywords = keywords + mesh_terms
            
            # Create Paper object
            paper = Paper(
                id=f"pmid:{pmid}" if pmid else None,
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=pub_date,
                doi=doi,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                source=PaperSource.PUBMED,
                journal=journal,
                volume=volume,
                issue=issue,
                pages=pages,
                keywords=all_keywords,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            return paper
            
        except Exception as e:
            logger.error(f"Error parsing PubMed article: {str(e)}")
            return None
    
    def _build_search_query(
        self,
        query: str,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        article_types: Optional[List[str]] = None
    ) -> str:
        """Build PubMed search query with filters."""
        search_parts = [query]
        
        # Add date filter
        if date_from or date_to:
            if date_from and date_to:
                date_filter = f"({date_from.strftime('%Y/%m/%d')}[PDAT]:{date_to.strftime('%Y/%m/%d')}[PDAT])"
            elif date_from:
                date_filter = f"{date_from.strftime('%Y/%m/%d')}[PDAT]:3000/12/31[PDAT]"
            elif date_to:
                date_filter = f"1800/01/01[PDAT]:{date_to.strftime('%Y/%m/%d')}[PDAT]"
            
            search_parts.append(date_filter)
        
        # Add article type filter
        if article_types:
            type_filters = []
            for article_type in article_types:
                type_filters.append(f"{article_type}[PT]")
            search_parts.append(f"({' OR '.join(type_filters)})")
        
        return " AND ".join(search_parts)
    
    async def get_paper_by_pmid(self, pmid: str) -> Optional[Paper]:
        """
        Get a specific paper by PMID.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            Paper object or None if not found
        """
        try:
            papers = await self._fetch_papers([pmid])
            return papers[0] if papers else None
            
        except Exception as e:
            logger.error(f"Error getting paper by PMID {pmid}: {str(e)}")
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
            search_query = f"{author_name}[AU]"
            return await self.search_papers(search_query, max_results)
            
        except Exception as e:
            logger.error(f"Error searching papers by author {author_name}: {str(e)}")
            return []
    
    def get_popular_article_types(self) -> Dict[str, str]:
        """Get popular PubMed article types."""
        return {
            "Journal Article": "Journal Article",
            "Review": "Review",
            "Clinical Trial": "Clinical Trial",
            "Randomized Controlled Trial": "Randomized Controlled Trial",
            "Meta-Analysis": "Meta-Analysis",
            "Systematic Review": "Systematic Review",
            "Case Reports": "Case Reports",
            "Letter": "Letter",
            "Editorial": "Editorial",
            "News": "News"
        }

# Global instance
pubmed_client = PubMedClient() 