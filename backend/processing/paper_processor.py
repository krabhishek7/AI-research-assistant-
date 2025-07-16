"""Paper processing module for extracting and cleaning paper content."""

import asyncio
import logging
import re
from typing import Optional, Dict, Any, List
from pathlib import Path
import tempfile
import os

import requests
import PyPDF2
import pdfplumber
from io import BytesIO

from backend.database.models import Paper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperProcessor:
    """Handles paper content extraction and text processing."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    async def extract_full_text(self, paper: Paper) -> Optional[str]:
        """
        Extract full text from a paper using PDF URL.
        
        Args:
            paper: Paper object containing metadata
            
        Returns:
            Extracted text or None if extraction fails
        """
        if not paper.pdf_url:
            logger.warning(f"No PDF URL available for paper: {paper.title}")
            return None
            
        try:
            # Download PDF content
            pdf_content = await self._download_pdf(paper.pdf_url)
            if not pdf_content:
                return None
                
            # Extract text using multiple methods
            text = self._extract_text_from_pdf(pdf_content)
            
            if text:
                # Clean and process the text
                cleaned_text = self._clean_text(text)
                return cleaned_text
                
            return None
            
        except Exception as e:
            logger.error(f"Error extracting text from {paper.title}: {str(e)}")
            return None
    
    async def _download_pdf(self, pdf_url: str) -> Optional[bytes]:
        """Download PDF content from URL."""
        try:
            response = self.session.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            # Check if the content is actually a PDF
            if response.headers.get('content-type', '').startswith('application/pdf'):
                return response.content
            else:
                logger.warning(f"URL does not return PDF content: {pdf_url}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {str(e)}")
            return None
    
    def _extract_text_from_pdf(self, pdf_content: bytes) -> Optional[str]:
        """Extract text from PDF using multiple methods."""
        # Try pdfplumber first (better formatting)
        try:
            text = self._extract_with_pdfplumber(pdf_content)
            if text and len(text.strip()) > 100:  # Ensure meaningful content
                return text
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")
        
        # Fallback to PyPDF2
        try:
            text = self._extract_with_pypdf2(pdf_content)
            if text and len(text.strip()) > 100:
                return text
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")
        
        return None
    
    def _extract_with_pdfplumber(self, pdf_content: bytes) -> Optional[str]:
        """Extract text using pdfplumber."""
        with BytesIO(pdf_content) as pdf_file:
            with pdfplumber.open(pdf_file) as pdf:
                text_parts = []
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                return '\n\n'.join(text_parts) if text_parts else None
    
    def _extract_with_pypdf2(self, pdf_content: bytes) -> Optional[str]:
        """Extract text using PyPDF2."""
        with BytesIO(pdf_content) as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            return '\n\n'.join(text_parts) if text_parts else None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\^\&\*\+\=\<\>\~\`]', '', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract common paper sections from text.
        
        Args:
            text: Full paper text
            
        Returns:
            Dictionary with section names as keys and content as values
        """
        sections = {
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'results': '',
            'discussion': '',
            'conclusion': '',
            'references': ''
        }
        
        if not text:
            return sections
        
        # Common section patterns
        section_patterns = {
            'abstract': r'(?i)(?:abstract|summary)\s*[:\-]?\s*\n(.*?)(?=\n\s*(?:introduction|keywords|1\.|i\.|background))',
            'introduction': r'(?i)(?:introduction|1\.|i\.)\s*(?:introduction)?\s*[:\-]?\s*\n(.*?)(?=\n\s*(?:methodology|method|2\.|ii\.|literature))',
            'methodology': r'(?i)(?:methodology|method|approach|2\.|ii\.)\s*[:\-]?\s*\n(.*?)(?=\n\s*(?:results|experiment|3\.|iii\.|evaluation))',
            'results': r'(?i)(?:results|findings|3\.|iii\.)\s*[:\-]?\s*\n(.*?)(?=\n\s*(?:discussion|conclusion|4\.|iv\.|analysis))',
            'discussion': r'(?i)(?:discussion|analysis|4\.|iv\.)\s*[:\-]?\s*\n(.*?)(?=\n\s*(?:conclusion|summary|5\.|v\.|references))',
            'conclusion': r'(?i)(?:conclusion|summary|5\.|v\.)\s*[:\-]?\s*\n(.*?)(?=\n\s*(?:references|bibliography|acknowledgment))',
            'references': r'(?i)(?:references|bibliography)\s*[:\-]?\s*\n(.*?)(?=\n\s*(?:appendix|$))'
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
        
        return sections
    
    def extract_key_information(self, text: str) -> Dict[str, Any]:
        """
        Extract key information from paper text.
        
        Args:
            text: Full paper text
            
        Returns:
            Dictionary with extracted information
        """
        info = {
            'word_count': 0,
            'key_terms': [],
            'figures_count': 0,
            'tables_count': 0,
            'equations_count': 0,
            'sections': {}
        }
        
        if not text:
            return info
        
        # Basic statistics
        info['word_count'] = len(text.split())
        
        # Count figures, tables, equations
        info['figures_count'] = len(re.findall(r'(?i)figure\s+\d+', text))
        info['tables_count'] = len(re.findall(r'(?i)table\s+\d+', text))
        info['equations_count'] = len(re.findall(r'(?i)equation\s+\d+', text))
        
        # Extract key terms (simple frequency-based approach)
        info['key_terms'] = self._extract_key_terms(text)
        
        # Extract sections
        info['sections'] = self.extract_sections(text)
        
        return info
    
    def _extract_key_terms(self, text: str, top_n: int = 20) -> List[str]:
        """Extract key terms from text using simple frequency analysis."""
        if not text:
            return []
        
        # Simple tokenization and frequency counting
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Common stop words to filter out
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our',
            'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
            'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'this', 'that', 'with',
            'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when',
            'come', 'here', 'into', 'like', 'long', 'look', 'make', 'many', 'over', 'such', 'take', 'than',
            'them', 'well', 'were', 'will', 'would', 'there', 'could', 'other', 'after', 'first', 'never',
            'these', 'think', 'where', 'being', 'every', 'great', 'might', 'shall', 'still', 'those', 'under',
            'while', 'should', 'another', 'between', 'through', 'during', 'before', 'without', 'around',
            'because', 'against', 'paper', 'using', 'based', 'approach', 'method', 'results', 'show', 'used',
            'work', 'data', 'model', 'also', 'may', 'proposed', 'studies', 'research', 'analysis', 'study'
        }
        
        # Filter and count
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count frequency
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top N
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, count in sorted_words[:top_n]]
    
    async def process_paper(self, paper: Paper) -> Paper:
        """
        Process a paper to extract and clean its content.
        
        Args:
            paper: Paper object to process
            
        Returns:
            Updated Paper object with processed content
        """
        try:
            # Extract full text if not already present
            if not paper.full_text:
                paper.full_text = await self.extract_full_text(paper)
            
            # Extract additional information if we have text
            if paper.full_text:
                key_info = self.extract_key_information(paper.full_text)
                
                # Update keywords if not already present
                if not paper.keywords and key_info['key_terms']:
                    paper.keywords = key_info['key_terms'][:10]  # Top 10 terms
            
            return paper
            
        except Exception as e:
            logger.error(f"Error processing paper {paper.title}: {str(e)}")
            return paper

# Global instance
paper_processor = PaperProcessor() 