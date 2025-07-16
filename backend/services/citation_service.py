"""
Citation Service
Provides comprehensive citation generation in multiple academic formats
"""

import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CitationStyle(Enum):
    """Supported citation styles"""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    IEEE = "ieee"
    VANCOUVER = "vancouver"
    BIBTEX = "bibtex"
    ENDNOTE = "endnote"
    ZOTERO = "zotero"

class PaperType(Enum):
    """Types of academic papers"""
    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    PREPRINT = "preprint"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    THESIS = "thesis"
    REPORT = "report"
    WEBSITE = "website"
    UNKNOWN = "unknown"

@dataclass
class Author:
    """Author information"""
    first_name: str
    last_name: str
    middle_name: Optional[str] = None
    suffix: Optional[str] = None
    
    def __post_init__(self):
        """Clean up author names"""
        self.first_name = self.first_name.strip()
        self.last_name = self.last_name.strip()
        if self.middle_name:
            self.middle_name = self.middle_name.strip()
        if self.suffix:
            self.suffix = self.suffix.strip()
    
    @property
    def full_name(self) -> str:
        """Get full name"""
        parts = [self.first_name]
        if self.middle_name:
            parts.append(self.middle_name)
        parts.append(self.last_name)
        if self.suffix:
            parts.append(self.suffix)
        return " ".join(parts)
    
    @property
    def last_first(self) -> str:
        """Get name in Last, First format"""
        name = self.last_name
        if self.first_name:
            name += f", {self.first_name}"
            if self.middle_name:
                name += f" {self.middle_name}"
        if self.suffix:
            name += f", {self.suffix}"
        return name
    
    @property
    def initials(self) -> str:
        """Get initials"""
        initials = []
        if self.first_name:
            initials.append(self.first_name[0].upper())
        if self.middle_name:
            initials.append(self.middle_name[0].upper())
        return ". ".join(initials) + "." if initials else ""

@dataclass
class Citation:
    """Citation data structure"""
    title: str
    authors: List[Author]
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    conference: Optional[str] = None
    institution: Optional[str] = None
    paper_type: PaperType = PaperType.UNKNOWN
    abstract: Optional[str] = None
    keywords: List[str] = None
    language: str = "en"
    accessed_date: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize defaults"""
        if self.keywords is None:
            self.keywords = []
        if self.accessed_date is None:
            self.accessed_date = datetime.now()

class CitationService:
    """
    Comprehensive citation service supporting multiple academic formats
    """
    
    def __init__(self):
        """Initialize citation service"""
        self.supported_styles = list(CitationStyle)
        logger.info(f"Citation service initialized with {len(self.supported_styles)} styles")
    
    def parse_authors(self, authors_input: Union[str, List[str]]) -> List[Author]:
        """
        Parse authors from various input formats
        
        Args:
            authors_input: String or list of author names
            
        Returns:
            List of Author objects
        """
        if not authors_input:
            return []
        
        # Convert to list if string
        if isinstance(authors_input, str):
            # Split by common separators
            authors_list = re.split(r'[,;]|\\band\\b|\\&', authors_input)
        else:
            authors_list = authors_input
        
        authors = []
        for author_str in authors_list:
            author_str = author_str.strip()
            if not author_str:
                continue
                
            # Parse different name formats
            author = self._parse_single_author(author_str)
            if author:
                authors.append(author)
        
        return authors
    
    def _parse_single_author(self, author_str: str) -> Optional[Author]:
        """Parse a single author name"""
        if not author_str:
            return None
        
        # Remove extra whitespace
        author_str = re.sub(r'\s+', ' ', author_str.strip())
        
        # Handle "Last, First Middle" format
        if ',' in author_str:
            parts = author_str.split(',', 1)
            last_name = parts[0].strip()
            first_part = parts[1].strip()
            
            # Split first part into first and middle names
            name_parts = first_part.split()
            first_name = name_parts[0] if name_parts else ""
            middle_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else None
            
            return Author(
                first_name=first_name,
                last_name=last_name,
                middle_name=middle_name
            )
        
        # Handle "First Middle Last" format
        else:
            parts = author_str.split()
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = parts[-1]
                middle_name = " ".join(parts[1:-1]) if len(parts) > 2 else None
                
                return Author(
                    first_name=first_name,
                    last_name=last_name,
                    middle_name=middle_name
                )
            elif len(parts) == 1:
                # Only one name, assume it's the last name
                return Author(
                    first_name="",
                    last_name=parts[0]
                )
        
        return None
    
    def detect_paper_type(self, paper_data: Dict[str, Any]) -> PaperType:
        """Detect the type of paper based on metadata"""
        
        # Check for journal indicators
        if paper_data.get('journal'):
            return PaperType.JOURNAL_ARTICLE
        
        # Check for conference indicators
        if paper_data.get('conference') or 'conference' in paper_data.get('title', '').lower():
            return PaperType.CONFERENCE_PAPER
        
        # Check for preprint indicators
        if paper_data.get('source') == 'arxiv' or 'arxiv' in paper_data.get('url', ''):
            return PaperType.PREPRINT
        
        # Check for book indicators
        if paper_data.get('publisher') and not paper_data.get('journal'):
            return PaperType.BOOK
        
        return PaperType.UNKNOWN
    
    def create_citation(self, paper_data: Dict[str, Any]) -> Citation:
        """
        Create a Citation object from paper data
        
        Args:
            paper_data: Dictionary containing paper metadata
            
        Returns:
            Citation object
        """
        
        # Parse authors
        authors = self.parse_authors(paper_data.get('authors', []))
        
        # Extract year from published date
        year = None
        if paper_data.get('published'):
            try:
                if isinstance(paper_data['published'], str):
                    # Try to parse date string
                    date_str = paper_data['published']
                    if re.match(r'^\d{4}$', date_str):
                        year = int(date_str)
                    elif re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                        year = int(date_str[:4])
                    else:
                        # Try to extract year from string
                        year_match = re.search(r'(\d{4})', date_str)
                        if year_match:
                            year = int(year_match.group(1))
                elif isinstance(paper_data['published'], int):
                    year = paper_data['published']
            except (ValueError, TypeError):
                pass
        
        # Detect paper type
        paper_type = self.detect_paper_type(paper_data)
        
        # Create citation
        citation = Citation(
            title=paper_data.get('title', 'Unknown Title'),
            authors=authors,
            year=year,
            doi=paper_data.get('doi'),
            url=paper_data.get('url'),
            journal=paper_data.get('journal'),
            volume=paper_data.get('volume'),
            issue=paper_data.get('issue'),
            pages=paper_data.get('pages'),
            publisher=paper_data.get('publisher'),
            conference=paper_data.get('conference'),
            institution=paper_data.get('institution'),
            paper_type=paper_type,
            abstract=paper_data.get('abstract'),
            keywords=paper_data.get('keywords', []),
            language=paper_data.get('language', 'en')
        )
        
        return citation
    
    def format_authors(self, authors: List[Author], style: CitationStyle, max_authors: int = None) -> str:
        """Format authors according to citation style"""
        if not authors:
            return "Unknown Author"
        
        if style == CitationStyle.APA:
            return self._format_authors_apa(authors, max_authors)
        elif style == CitationStyle.MLA:
            return self._format_authors_mla(authors, max_authors)
        elif style == CitationStyle.CHICAGO:
            return self._format_authors_chicago(authors, max_authors)
        elif style == CitationStyle.HARVARD:
            return self._format_authors_harvard(authors, max_authors)
        elif style == CitationStyle.IEEE:
            return self._format_authors_ieee(authors, max_authors)
        elif style == CitationStyle.VANCOUVER:
            return self._format_authors_vancouver(authors, max_authors)
        else:
            return self._format_authors_default(authors, max_authors)
    
    def _format_authors_apa(self, authors: List[Author], max_authors: int = None) -> str:
        """Format authors in APA style"""
        if not authors:
            return "Unknown Author"
        
        max_authors = max_authors or 7  # APA 7th edition rule
        
        if len(authors) == 1:
            author = authors[0]
            return f"{author.last_name}, {author.initials}"
        elif len(authors) == 2:
            return f"{authors[0].last_name}, {authors[0].initials}, & {authors[1].last_name}, {authors[1].initials}"
        elif len(authors) <= max_authors:
            formatted = []
            for i, author in enumerate(authors):
                if i == len(authors) - 1:  # Last author
                    formatted.append(f"& {author.last_name}, {author.initials}")
                else:
                    formatted.append(f"{author.last_name}, {author.initials}")
            return ", ".join(formatted)
        else:
            # More than max_authors, use et al.
            first_author = authors[0]
            return f"{first_author.last_name}, {first_author.initials}, et al."
    
    def _format_authors_mla(self, authors: List[Author], max_authors: int = None) -> str:
        """Format authors in MLA style"""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            author = authors[0]
            return f"{author.last_name}, {author.first_name}"
        elif len(authors) == 2:
            return f"{authors[0].last_name}, {authors[0].first_name}, and {authors[1].first_name} {authors[1].last_name}"
        else:
            first_author = authors[0]
            return f"{first_author.last_name}, {first_author.first_name}, et al."
    
    def _format_authors_chicago(self, authors: List[Author], max_authors: int = None) -> str:
        """Format authors in Chicago style"""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            author = authors[0]
            return f"{author.last_name}, {author.first_name}"
        elif len(authors) <= 3:
            formatted = []
            for i, author in enumerate(authors):
                if i == 0:
                    formatted.append(f"{author.last_name}, {author.first_name}")
                elif i == len(authors) - 1:
                    formatted.append(f"and {author.first_name} {author.last_name}")
                else:
                    formatted.append(f"{author.first_name} {author.last_name}")
            return ", ".join(formatted)
        else:
            first_author = authors[0]
            return f"{first_author.last_name}, {first_author.first_name}, et al."
    
    def _format_authors_harvard(self, authors: List[Author], max_authors: int = None) -> str:
        """Format authors in Harvard style"""
        return self._format_authors_apa(authors, max_authors)  # Similar to APA
    
    def _format_authors_ieee(self, authors: List[Author], max_authors: int = None) -> str:
        """Format authors in IEEE style"""
        if not authors:
            return "Unknown Author"
        
        if len(authors) <= 3:
            formatted = []
            for author in authors:
                formatted.append(f"{author.first_name[0]}. {author.last_name}" if author.first_name else author.last_name)
            return ", ".join(formatted)
        else:
            first_author = authors[0]
            return f"{first_author.first_name[0]}. {first_author.last_name} et al." if first_author.first_name else f"{first_author.last_name} et al."
    
    def _format_authors_vancouver(self, authors: List[Author], max_authors: int = None) -> str:
        """Format authors in Vancouver style"""
        if not authors:
            return "Unknown Author"
        
        if len(authors) <= 6:
            formatted = []
            for author in authors:
                formatted.append(f"{author.last_name} {author.initials}")
            return ", ".join(formatted)
        else:
            # First 3 authors, then et al.
            formatted = []
            for i, author in enumerate(authors[:3]):
                formatted.append(f"{author.last_name} {author.initials}")
            return ", ".join(formatted) + ", et al."
    
    def _format_authors_default(self, authors: List[Author], max_authors: int = None) -> str:
        """Default author formatting"""
        if not authors:
            return "Unknown Author"
        
        if len(authors) == 1:
            return authors[0].full_name
        elif len(authors) <= 3:
            names = [author.full_name for author in authors]
            return ", ".join(names[:-1]) + f" and {names[-1]}"
        else:
            return f"{authors[0].full_name} et al."
    
    def generate_citation(self, citation: Citation, style: CitationStyle) -> str:
        """
        Generate a citation string in the specified style
        
        Args:
            citation: Citation object
            style: Citation style
            
        Returns:
            Formatted citation string
        """
        
        if style == CitationStyle.APA:
            return self._generate_apa_citation(citation)
        elif style == CitationStyle.MLA:
            return self._generate_mla_citation(citation)
        elif style == CitationStyle.CHICAGO:
            return self._generate_chicago_citation(citation)
        elif style == CitationStyle.HARVARD:
            return self._generate_harvard_citation(citation)
        elif style == CitationStyle.IEEE:
            return self._generate_ieee_citation(citation)
        elif style == CitationStyle.VANCOUVER:
            return self._generate_vancouver_citation(citation)
        elif style == CitationStyle.BIBTEX:
            return self._generate_bibtex_citation(citation)
        else:
            return self._generate_default_citation(citation)
    
    def _generate_apa_citation(self, citation: Citation) -> str:
        """Generate APA style citation"""
        parts = []
        
        # Authors
        authors_str = self.format_authors(citation.authors, CitationStyle.APA)
        parts.append(authors_str)
        
        # Year
        year_str = f"({citation.year})" if citation.year else "(n.d.)"
        parts.append(year_str)
        
        # Title
        title_str = f"{citation.title}."
        parts.append(title_str)
        
        # Journal/Source
        if citation.journal:
            journal_str = f"*{citation.journal}*"
            if citation.volume:
                journal_str += f", {citation.volume}"
                if citation.issue:
                    journal_str += f"({citation.issue})"
            if citation.pages:
                journal_str += f", {citation.pages}"
            journal_str += "."
            parts.append(journal_str)
        
        # DOI or URL
        if citation.doi:
            parts.append(f"https://doi.org/{citation.doi}")
        elif citation.url:
            parts.append(citation.url)
        
        return " ".join(parts)
    
    def _generate_mla_citation(self, citation: Citation) -> str:
        """Generate MLA style citation"""
        parts = []
        
        # Authors
        authors_str = self.format_authors(citation.authors, CitationStyle.MLA)
        parts.append(f"{authors_str}.")
        
        # Title
        title_str = f'"{citation.title}."'
        parts.append(title_str)
        
        # Journal/Source
        if citation.journal:
            journal_str = f"*{citation.journal}*"
            if citation.volume:
                journal_str += f", vol. {citation.volume}"
                if citation.issue:
                    journal_str += f", no. {citation.issue}"
            if citation.year:
                journal_str += f", {citation.year}"
            if citation.pages:
                journal_str += f", pp. {citation.pages}"
            journal_str += "."
            parts.append(journal_str)
        
        # URL
        if citation.url:
            parts.append(f"Web. {citation.accessed_date.strftime('%d %b %Y')}.")
        
        return " ".join(parts)
    
    def _generate_chicago_citation(self, citation: Citation) -> str:
        """Generate Chicago style citation"""
        parts = []
        
        # Authors
        authors_str = self.format_authors(citation.authors, CitationStyle.CHICAGO)
        parts.append(f"{authors_str}.")
        
        # Title
        title_str = f'"{citation.title}."'
        parts.append(title_str)
        
        # Journal/Source
        if citation.journal:
            journal_str = f"*{citation.journal}*"
            if citation.volume:
                journal_str += f" {citation.volume}"
                if citation.issue:
                    journal_str += f", no. {citation.issue}"
            if citation.year:
                journal_str += f" ({citation.year})"
            if citation.pages:
                journal_str += f": {citation.pages}"
            journal_str += "."
            parts.append(journal_str)
        
        # URL
        if citation.url:
            parts.append(f"Accessed {citation.accessed_date.strftime('%B %d, %Y')}. {citation.url}.")
        
        return " ".join(parts)
    
    def _generate_harvard_citation(self, citation: Citation) -> str:
        """Generate Harvard style citation"""
        # Harvard is similar to APA
        return self._generate_apa_citation(citation)
    
    def _generate_ieee_citation(self, citation: Citation) -> str:
        """Generate IEEE style citation"""
        parts = []
        
        # Authors
        authors_str = self.format_authors(citation.authors, CitationStyle.IEEE)
        parts.append(f"{authors_str},")
        
        # Title
        title_str = f'"{citation.title},"'
        parts.append(title_str)
        
        # Journal/Source
        if citation.journal:
            journal_str = f"*{citation.journal}*"
            if citation.volume:
                journal_str += f", vol. {citation.volume}"
                if citation.issue:
                    journal_str += f", no. {citation.issue}"
            if citation.pages:
                journal_str += f", pp. {citation.pages}"
            if citation.year:
                journal_str += f", {citation.year}"
            journal_str += "."
            parts.append(journal_str)
        
        return " ".join(parts)
    
    def _generate_vancouver_citation(self, citation: Citation) -> str:
        """Generate Vancouver style citation"""
        parts = []
        
        # Authors
        authors_str = self.format_authors(citation.authors, CitationStyle.VANCOUVER)
        parts.append(f"{authors_str}.")
        
        # Title
        title_str = f"{citation.title}."
        parts.append(title_str)
        
        # Journal/Source
        if citation.journal:
            journal_str = f"{citation.journal}."
            if citation.year:
                journal_str += f" {citation.year}"
            if citation.volume:
                journal_str += f";{citation.volume}"
                if citation.issue:
                    journal_str += f"({citation.issue})"
            if citation.pages:
                journal_str += f":{citation.pages}"
            journal_str += "."
            parts.append(journal_str)
        
        return " ".join(parts)
    
    def _generate_bibtex_citation(self, citation: Citation) -> str:
        """Generate BibTeX citation"""
        
        # Determine entry type
        entry_type = "article"
        if citation.paper_type == PaperType.CONFERENCE_PAPER:
            entry_type = "inproceedings"
        elif citation.paper_type == PaperType.BOOK:
            entry_type = "book"
        elif citation.paper_type == PaperType.BOOK_CHAPTER:
            entry_type = "incollection"
        elif citation.paper_type == PaperType.THESIS:
            entry_type = "phdthesis"
        elif citation.paper_type == PaperType.PREPRINT:
            entry_type = "misc"
        
        # Generate citation key
        first_author = citation.authors[0] if citation.authors else None
        key_parts = []
        if first_author:
            key_parts.append(first_author.last_name.lower())
        if citation.year:
            key_parts.append(str(citation.year))
        citation_key = "_".join(key_parts) if key_parts else "unknown"
        
        # Build BibTeX entry
        lines = [f"@{entry_type}{{{citation_key},"]
        
        # Add fields
        lines.append(f'  title = {{{citation.title}}},')
        
        if citation.authors:
            authors_str = " and ".join([author.full_name for author in citation.authors])
            lines.append(f'  author = {{{authors_str}}},')
        
        if citation.year:
            lines.append(f'  year = {{{citation.year}}},')
        
        if citation.journal:
            lines.append(f'  journal = {{{citation.journal}}},')
        
        if citation.volume:
            lines.append(f'  volume = {{{citation.volume}}},')
        
        if citation.issue:
            lines.append(f'  number = {{{citation.issue}}},')
        
        if citation.pages:
            lines.append(f'  pages = {{{citation.pages}}},')
        
        if citation.doi:
            lines.append(f'  doi = {{{citation.doi}}},')
        
        if citation.url:
            lines.append(f'  url = {{{citation.url}}},')
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _generate_default_citation(self, citation: Citation) -> str:
        """Generate default citation format"""
        parts = []
        
        # Authors
        authors_str = self.format_authors(citation.authors, CitationStyle.APA)
        parts.append(authors_str)
        
        # Year
        if citation.year:
            parts.append(f"({citation.year})")
        
        # Title
        parts.append(f"{citation.title}.")
        
        # Journal
        if citation.journal:
            parts.append(f"*{citation.journal}*.")
        
        # URL
        if citation.url:
            parts.append(f"Retrieved from {citation.url}")
        
        return " ".join(parts)
    
    def export_citations(self, citations: List[Citation], style: CitationStyle, format_type: str = "text") -> str:
        """
        Export multiple citations
        
        Args:
            citations: List of Citation objects
            style: Citation style
            format_type: Export format (text, json, xml)
            
        Returns:
            Formatted citations string
        """
        
        if format_type == "json":
            return self._export_citations_json(citations)
        elif format_type == "xml":
            return self._export_citations_xml(citations)
        else:
            # Text format
            formatted_citations = []
            for i, citation in enumerate(citations, 1):
                formatted_citation = self.generate_citation(citation, style)
                formatted_citations.append(f"{i}. {formatted_citation}")
            return "\n\n".join(formatted_citations)
    
    def _export_citations_json(self, citations: List[Citation]) -> str:
        """Export citations as JSON"""
        citations_data = []
        for citation in citations:
            citation_data = {
                "title": citation.title,
                "authors": [author.full_name for author in citation.authors],
                "year": citation.year,
                "journal": citation.journal,
                "doi": citation.doi,
                "url": citation.url,
                "paper_type": citation.paper_type.value
            }
            citations_data.append(citation_data)
        
        return json.dumps(citations_data, indent=2)
    
    def _export_citations_xml(self, citations: List[Citation]) -> str:
        """Export citations as XML"""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<citations>']
        
        for citation in citations:
            lines.append('  <citation>')
            lines.append(f'    <title>{citation.title}</title>')
            lines.append('    <authors>')
            for author in citation.authors:
                lines.append(f'      <author>{author.full_name}</author>')
            lines.append('    </authors>')
            if citation.year:
                lines.append(f'    <year>{citation.year}</year>')
            if citation.journal:
                lines.append(f'    <journal>{citation.journal}</journal>')
            if citation.doi:
                lines.append(f'    <doi>{citation.doi}</doi>')
            if citation.url:
                lines.append(f'    <url>{citation.url}</url>')
            lines.append('  </citation>')
        
        lines.append('</citations>')
        return '\n'.join(lines)
    
    def get_supported_styles(self) -> List[str]:
        """Get list of supported citation styles"""
        return [style.value for style in CitationStyle]
    
    def validate_citation(self, citation: Citation) -> Dict[str, Any]:
        """
        Validate citation data and return validation results
        
        Args:
            citation: Citation object to validate
            
        Returns:
            Dictionary with validation results
        """
        
        issues = []
        warnings = []
        
        # Check required fields
        if not citation.title or citation.title.strip() == "Unknown Title":
            issues.append("Missing or invalid title")
        
        if not citation.authors:
            issues.append("Missing authors")
        
        if not citation.year:
            warnings.append("Missing publication year")
        
        # Check author names
        for i, author in enumerate(citation.authors):
            if not author.last_name:
                issues.append(f"Author {i+1} missing last name")
            if not author.first_name:
                warnings.append(f"Author {i+1} missing first name")
        
        # Check journal articles
        if citation.paper_type == PaperType.JOURNAL_ARTICLE:
            if not citation.journal:
                warnings.append("Journal article missing journal name")
        
        # Check DOI format
        if citation.doi and not re.match(r'^10\.\d+/', citation.doi):
            warnings.append("DOI format appears invalid")
        
        # Check URL format
        if citation.url and not re.match(r'^https?://', citation.url):
            warnings.append("URL format appears invalid")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "quality_score": max(0, 100 - len(issues) * 25 - len(warnings) * 5)
        }

# Global citation service instance
_citation_service = None

def get_citation_service() -> CitationService:
    """
    Get the global citation service instance
    
    Returns:
        CitationService instance
    """
    global _citation_service
    if _citation_service is None:
        _citation_service = CitationService()
    return _citation_service

def generate_citation_from_paper(paper_data: Dict[str, Any], style: str = "apa") -> Dict[str, Any]:
    """
    Convenience function to generate citation from paper data
    
    Args:
        paper_data: Dictionary containing paper metadata
        style: Citation style string
        
    Returns:
        Dictionary with citation results
    """
    try:
        citation_service = get_citation_service()
        
        # Parse style
        try:
            citation_style = CitationStyle(style.lower())
        except ValueError:
            citation_style = CitationStyle.APA
        
        # Create citation
        citation = citation_service.create_citation(paper_data)
        
        # Generate citation string
        citation_string = citation_service.generate_citation(citation, citation_style)
        
        # Validate citation
        validation = citation_service.validate_citation(citation)
        
        return {
            "citation": citation_string,
            "style": citation_style.value,
            "validation": validation,
            "authors_count": len(citation.authors),
            "paper_type": citation.paper_type.value
        }
        
    except Exception as e:
        logger.error(f"Error generating citation: {e}")
        return {
            "citation": "Error generating citation",
            "style": style,
            "validation": {"valid": False, "issues": [str(e)], "warnings": []},
            "authors_count": 0,
            "paper_type": "unknown"
        } 