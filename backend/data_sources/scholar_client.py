"""Google Scholar client for fetching academic papers."""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path
import re
import random
from urllib.parse import quote_plus, urljoin

import aiohttp
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.database.models import Paper, Author, PaperSource
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleScholarClient:
    """Client for scraping Google Scholar (respecting rate limits and terms of service)."""
    
    def __init__(self):
        self.base_url = "https://scholar.google.com"
        self.search_url = f"{self.base_url}/scholar"
        self.rate_limit = 2  # seconds between requests (conservative)
        self.session = None
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with headers."""
        if self.session is None:
            headers = {
                "User-Agent": random.choice(self.user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
            self.session = aiohttp.ClientSession(headers=headers)
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
        include_patents: bool = False,
        include_citations: bool = False
    ) -> List[Paper]:
        """
        Search for papers on Google Scholar.
        """
        try:
            logger.info(f"Searching Google Scholar with query: {query}")
            
            # Add timeout to prevent hanging
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            
            # Build search parameters
            params = self._build_search_params(
                query, max_results, sort, date_from, date_to, include_patents, include_citations
            )
            
            # Search Google Scholar with timeout
            papers = await asyncio.wait_for(
                self._search_scholar(params, max_results), 
                timeout=15.0  # 15 second timeout
            )
            
            logger.info(f"Found {len(papers)} papers from Google Scholar")
            return papers
            
        except asyncio.TimeoutError:
            logger.warning("Google Scholar search timed out")
            return []
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {str(e)}")
            return []  # Return empty list instead of raising
    
    async def _search_scholar(self, params: Dict[str, Any], max_results: int) -> List[Paper]:
        """Search Google Scholar with the given parameters."""
        session = await self._get_session()
        papers = []
        
        # Google Scholar shows 10 results per page
        results_per_page = 10
        pages_needed = (max_results + results_per_page - 1) // results_per_page
        
        for page in range(pages_needed):
            if page > 0:
                params['start'] = page * results_per_page
            
            try:
                logger.info(f"Fetching page {page} with params: {params}")
                async with session.get(self.search_url, params=params) as response:
                    logger.info(f"Response status: {response.status}")
                    
                    if response.status == 429:
                        logger.warning("Rate limited by Google Scholar, waiting...")
                        await asyncio.sleep(self.rate_limit * 3)
                        continue
                    
                    response.raise_for_status()
                    html_content = await response.text()
                    
                    logger.info(f"HTML content length: {len(html_content)}")
                    
                    # Parse the HTML
                    page_papers = self._parse_scholar_html(html_content)
                    papers.extend(page_papers)
                    
                    logger.info(f"Parsed {len(page_papers)} papers from page {page}")
                    
                    # Check if we have enough results
                    if len(papers) >= max_results:
                        break
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit)
                    
                    with open(f'scholar_debug_page_{page}.html', 'w') as f:
                        f.write(html_content)
                    logger.info(f"Saved HTML for page {page} to scholar_debug_page_{page}.html")
                    
            except Exception as e:
                logger.error(f"Error fetching page {page}: {str(e)}")
                import traceback
                traceback.print_exc()
                break
        
        return papers[:max_results]
    
    def _parse_scholar_html(self, html_content: str) -> List[Paper]:
        papers = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Save debug HTML first
            with open('scholar_full_debug.html', 'w') as f:
                f.write(html_content)
            logger.info("Saved full HTML to scholar_full_debug.html")
            
            # Try multiple selectors for Google Scholar results
            result_containers = []
            
            # Try different selectors in order of preference
            selectors = [
                'div.gs_r.gs_or.gs_scl[data-cid]',  # Current selector
                'div.gs_r.gs_or.gs_scl',            # Without data-cid
                'div.gs_r',                         # Basic selector
                'div[data-cid]',                    # Any div with data-cid
                '.gs_r'                             # Class only
            ]
            
            for selector in selectors:
                result_containers = soup.select(selector)
                if result_containers:
                    logger.info(f"Found {len(result_containers)} containers with selector: {selector}")
                    break
                else:
                    logger.debug(f"No containers found with selector: {selector}")
            
            if not result_containers:
                logger.warning("No result containers found with any selector")
                # Check for common blocking patterns
                if "captcha" in html_content.lower() or "unusual traffic" in html_content.lower():
                    logger.error("CAPTCHA detected in response")
                elif "robot" in html_content.lower():
                    logger.error("Robot detection message found")
                elif len(html_content) < 1000:
                    logger.error(f"Response too short ({len(html_content)} chars), likely blocked")
                else:
                    logger.error("Unknown parsing issue - check scholar_full_debug.html")
                return papers
            
            # Parse each container
            for i, container in enumerate(result_containers):
                logger.debug(f"Processing container {i+1}/{len(result_containers)}")
                paper = self._parse_single_result(container)
                if paper:
                    papers.append(paper)
                    logger.debug(f"Successfully parsed paper: {paper.title}")
                else:
                    logger.debug(f"Failed to parse container {i+1}")
            
            logger.info(f"Successfully parsed {len(papers)} papers from {len(result_containers)} containers")
            
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return papers

    def _parse_single_result(self, container) -> Optional[Paper]:
        try:
            title_elem = container.select_one('h3.gs_rt a')
            if not title_elem:
                title_elem = container.select_one('h3.gs_rt')
            if not title_elem:
                return None
            title = title_elem.get_text(strip=True)
            title = re.sub(r'\[(PDF|HTML|CITATION)\]', '', title).strip()
            if not title:
                return None
            link_elem = title_elem if title_elem.name == 'a' else title_elem.find('a')
            url = link_elem.get('href') if link_elem else None
            if url and url.startswith('/'):
                url = urljoin(self.base_url, url)
            info_elem = container.select_one('div.gs_a')
            authors = []
            publication_info = ""
            journal = None
            year = None
            if info_elem:
                info_text = info_elem.get_text(strip=True)
                publication_info = info_text
                parts = re.split(r'\s*-\s*', info_text)
                if parts:
                    author_text = parts[0].strip()
                    author_names = re.split(r',\s*|\s+and\s+|\s*&\s*', author_text)
                    authors = [Author(name=name.strip()) for name in author_names if name.strip() and not re.search(r'\d', name)]
                if len(parts) > 1:
                    pub_part = parts[1].strip()
                    year_match = re.search(r'(\d{4})', pub_part)
                    year = int(year_match.group(1)) if year_match else None
                    journal = re.sub(r',?\s*\d{4}.*$', '', pub_part).strip()
            abstract_elem = container.select_one('div.gs_rs')
            abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""
            cited_elem = container.select_one('a[href*="cites="]')
            cited_by = int(re.search(r'(\d+)', cited_elem.get_text()).group(1)) if cited_elem and re.search(r'(\d+)', cited_elem.get_text()) else None
            import uuid
            paper = Paper(
                id=f"scholar:{uuid.uuid4().hex[:16]}",
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=datetime(year, 1, 1) if year else None,
                url=url,
                source=PaperSource.GOOGLE_SCHOLAR,
                journal=journal,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            return paper
        except Exception as e:
            logger.error(f"Error parsing result: {str(e)}")
            return None
    
    def _build_search_params(
        self,
        query: str,
        max_results: int,
        sort: str,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        include_patents: bool = False,
        include_citations: bool = False
    ) -> Dict[str, Any]:
        """Build Google Scholar search parameters."""
        params = {
            'q': query,
            'hl': 'en',
            'as_sdt': '0,5'  # Include both articles and patents
        }
        
        # Sort order
        if sort == "date":
            params['scisbd'] = '1'  # Sort by date
        
        # Date filtering
        if date_from or date_to:
            if date_from:
                params['as_ylo'] = date_from.year
            if date_to:
                params['as_yhi'] = date_to.year
        
        # Exclude patents if requested
        if not include_patents:
            params['as_sdt'] = '0'
        
        # Include citations if requested
        if include_citations:
            params['scisbd'] = '2'
        
        return params
    
    async def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """
        Get detailed information about a paper.
        This is limited by Google Scholar's structure.
        
        Args:
            paper_id: Paper ID (usually hash-based for Scholar)
            
        Returns:
            Paper object or None if not found
        """
        try:
            # For Google Scholar, we can't easily get individual paper details
            # without the original search context, so this is a placeholder
            logger.warning("Google Scholar doesn't support individual paper lookup")
            return None
            
        except Exception as e:
            logger.error(f"Error getting paper details: {str(e)}")
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
            search_query = f'author:"{author_name}"'
            return await self.search_papers(search_query, max_results)
            
        except Exception as e:
            logger.error(f"Error searching papers by author {author_name}: {str(e)}")
            return []
    
    def get_search_tips(self) -> Dict[str, str]:
        """Get Google Scholar search tips."""
        return {
            "author": 'author:"John Smith" - Search by author name',
            "title": 'intitle:"machine learning" - Search in title',
            "publication": 'source:"Nature" - Search in specific publication',
            "date": 'after:2020 - Papers after 2020',
            "filetype": 'filetype:pdf - Find PDF files',
            "exact": '"exact phrase" - Search exact phrase',
            "exclude": '-excluded - Exclude term',
            "OR": 'term1 OR term2 - Either term',
            "site": 'site:arxiv.org - Search specific site'
        }

# Global instance
google_scholar_client = GoogleScholarClient() 