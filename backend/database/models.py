from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class PaperSource(str, Enum):
    """Enumeration of paper sources."""
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    GOOGLE_SCHOLAR = "google_scholar"
    MANUAL = "manual"

class CitationStyle(str, Enum):
    """Enumeration of citation styles."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    BIBTEX = "bibtex"

class Author(BaseModel):
    """Model for paper authors."""
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None

class Paper(BaseModel):
    """Main paper model with all metadata."""
    
    # Core identification
    id: Optional[str] = None
    title: str
    authors: List[Author]
    abstract: str
    
    # Publication details
    publication_date: Optional[datetime] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    
    # Identifiers
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pubmed_id: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    
    # Content
    full_text: Optional[str] = None
    
    # Metadata
    source: PaperSource
    categories: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    
    # Metrics
    citation_count: Optional[int] = None
    impact_factor: Optional[float] = None
    
    # Processing metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    embedding_generated: bool = False
    
    # Search relevance (populated during search)
    relevance_score: Optional[float] = None
    
    @validator('authors', pre=True)
    def validate_authors(cls, v):
        """Ensure authors is a list of Author objects."""
        if not v:
            return []
        
        result = []
        for author in v:
            if isinstance(author, dict):
                result.append(Author(**author))
            elif isinstance(author, str):
                result.append(Author(name=author))
            else:
                result.append(author)
        return result

class SearchQuery(BaseModel):
    """Model for search queries."""
    query: str
    max_results: int = Field(10, le=50)
    sources: List[PaperSource] = Field(default_factory=lambda: list(PaperSource))
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    categories: List[str] = Field(default_factory=list)
    
class SearchResult(BaseModel):
    """Model for search results."""
    papers: List[Paper]
    total_results: int
    query: str
    search_time: float
    sources_searched: List[PaperSource]

class PaperSummary(BaseModel):
    """Model for paper summaries."""
    paper_id: str
    title: str
    summary: str
    key_points: List[str]
    methodology: Optional[str] = None
    findings: Optional[str] = None
    limitations: Optional[str] = None
    future_work: Optional[str] = None
    confidence_score: Optional[float] = None
    generated_at: datetime = Field(default_factory=datetime.now)

class Citation(BaseModel):
    """Model for formatted citations."""
    paper_id: str
    style: CitationStyle
    formatted_citation: str
    in_text_citation: str
    generated_at: datetime = Field(default_factory=datetime.now)

class ResearchProject(BaseModel):
    """Model for organizing papers into research projects."""
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    papers: List[str] = Field(default_factory=list)  # List of paper IDs
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class APIResponse(BaseModel):
    """Standard API response model."""
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now) 