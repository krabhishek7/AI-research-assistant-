"""
Export Service
Provides functionality to export summaries, citations, reading lists, and other data
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import json
import csv
import io
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.database.models import Paper, Author, PaperSource
from backend.services.citation_service import get_citation_service, CitationStyle
from backend.processing.summarizer import get_summarizer
from backend.services.recommendation_service import recommendation_service
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExportFormat:
    """Supported export formats"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "md"
    BIBTEX = "bib"
    ENDNOTE = "enw"
    RIS = "ris"
    PDF = "pdf"  # Future implementation

class ExportService:
    """
    Comprehensive export service for academic data
    """
    
    def __init__(self):
        self.citation_service = get_citation_service()
        self.summarizer = get_summarizer()
        self.recommendation_service = recommendation_service
        self.supported_formats = [
            ExportFormat.JSON,
            ExportFormat.CSV,
            ExportFormat.XML,
            ExportFormat.TXT,
            ExportFormat.HTML,
            ExportFormat.MARKDOWN,
            ExportFormat.BIBTEX,
            ExportFormat.ENDNOTE,
            ExportFormat.RIS
        ]
        logger.info("Export service initialized")
    
    async def export_papers(
        self,
        papers: List[Paper],
        export_format: str,
        include_abstracts: bool = True,
        include_citations: bool = True,
        citation_style: str = "apa",
        include_summaries: bool = False,
        custom_fields: List[str] = None
    ) -> str:
        """
        Export a list of papers in the specified format
        
        Args:
            papers: List of papers to export
            export_format: Export format (json, csv, xml, txt, html, md, bib, enw, ris)
            include_abstracts: Whether to include abstracts
            include_citations: Whether to include formatted citations
            citation_style: Citation style to use
            include_summaries: Whether to include AI-generated summaries
            custom_fields: Custom fields to include
            
        Returns:
            Exported data as string
        """
        try:
            logger.info(f"Exporting {len(papers)} papers to {export_format}")
            
            # Prepare data with optional fields
            export_data = await self._prepare_export_data(
                papers,
                include_abstracts,
                include_citations,
                citation_style,
                include_summaries,
                custom_fields
            )
            
            # Export based on format
            if export_format == ExportFormat.JSON:
                return self._export_json(export_data)
            elif export_format == ExportFormat.CSV:
                return self._export_csv(export_data)
            elif export_format == ExportFormat.XML:
                return self._export_xml(export_data)
            elif export_format == ExportFormat.TXT:
                return self._export_txt(export_data)
            elif export_format == ExportFormat.HTML:
                return self._export_html(export_data)
            elif export_format == ExportFormat.MARKDOWN:
                return self._export_markdown(export_data)
            elif export_format == ExportFormat.BIBTEX:
                return self._export_bibtex(papers)
            elif export_format == ExportFormat.ENDNOTE:
                return self._export_endnote(papers)
            elif export_format == ExportFormat.RIS:
                return self._export_ris(papers)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            logger.error(f"Error exporting papers: {str(e)}")
            raise
    
    async def export_citations(
        self,
        papers: List[Paper],
        citation_style: str = "apa",
        export_format: str = ExportFormat.TXT,
        include_metadata: bool = True
    ) -> str:
        """
        Export citations for a list of papers
        
        Args:
            papers: List of papers to cite
            citation_style: Citation style (apa, mla, chicago, etc.)
            export_format: Export format for citations
            include_metadata: Whether to include metadata
            
        Returns:
            Exported citations as string
        """
        try:
            logger.info(f"Exporting {len(papers)} citations in {citation_style} style")
            
            citations = []
            
            for paper in papers:
                # Generate citation
                paper_data = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "published": paper.publication_date.isoformat() if paper.publication_date else None,
                    "journal": getattr(paper, 'journal', None),
                    "url": paper.url,
                    "doi": getattr(paper, 'doi', None),
                    "abstract": paper.abstract,
                    "source": paper.source.value if paper.source else None
                }
                
                citation_result = self.citation_service.generate_citation_from_paper(
                    paper_data, citation_style
                )
                
                citation_entry = {
                    "paper_id": paper.id,
                    "title": paper.title,
                    "citation": citation_result["citation"],
                    "style": citation_result["style"],
                    "paper_type": citation_result["paper_type"],
                    "authors_count": citation_result["authors_count"]
                }
                
                if include_metadata:
                    citation_entry.update({
                        "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                        "journal": getattr(paper, 'journal', None),
                        "doi": getattr(paper, 'doi', None),
                        "url": paper.url,
                        "source": paper.source.value if paper.source else None
                    })
                
                citations.append(citation_entry)
            
            # Export citations
            if export_format == ExportFormat.JSON:
                return self._export_json(citations)
            elif export_format == ExportFormat.CSV:
                return self._export_csv(citations)
            elif export_format == ExportFormat.TXT:
                return self._export_citations_txt(citations)
            elif export_format == ExportFormat.HTML:
                return self._export_citations_html(citations)
            elif export_format == ExportFormat.MARKDOWN:
                return self._export_citations_markdown(citations)
            else:
                return self._export_citations_txt(citations)
                
        except Exception as e:
            logger.error(f"Error exporting citations: {str(e)}")
            raise
    
    async def export_reading_list(
        self,
        user_id: str,
        export_format: str = ExportFormat.HTML,
        include_summaries: bool = True,
        include_notes: bool = True,
        include_ratings: bool = True
    ) -> str:
        """
        Export user's reading list
        
        Args:
            user_id: User ID
            export_format: Export format
            include_summaries: Whether to include summaries
            include_notes: Whether to include notes
            include_ratings: Whether to include ratings
            
        Returns:
            Exported reading list as string
        """
        try:
            logger.info(f"Exporting reading list for user {user_id}")
            
            # Get user profile
            user_profile = self.recommendation_service.get_user_profile(user_id)
            
            # Get reading list data
            reading_list_data = {
                "user_id": user_id,
                "generated_at": datetime.now().isoformat(),
                "total_papers": len(user_profile.reading_history),
                "total_bookmarks": len(user_profile.bookmarks),
                "reading_history": user_profile.reading_history,
                "bookmarks": user_profile.bookmarks,
                "ratings": user_profile.ratings if include_ratings else {},
                "reading_time": user_profile.reading_time,
                "favorite_authors": user_profile.favorite_authors,
                "favorite_journals": user_profile.favorite_journals,
                "favorite_categories": user_profile.favorite_categories
            }
            
            # Export based on format
            if export_format == ExportFormat.JSON:
                return self._export_json(reading_list_data)
            elif export_format == ExportFormat.HTML:
                return self._export_reading_list_html(reading_list_data)
            elif export_format == ExportFormat.MARKDOWN:
                return self._export_reading_list_markdown(reading_list_data)
            elif export_format == ExportFormat.TXT:
                return self._export_reading_list_txt(reading_list_data)
            else:
                return self._export_json(reading_list_data)
                
        except Exception as e:
            logger.error(f"Error exporting reading list: {str(e)}")
            raise
    
    async def export_summaries(
        self,
        papers: List[Paper],
        export_format: str = ExportFormat.HTML,
        summary_method: str = "abstractive",
        include_key_findings: bool = True,
        max_length: int = 500
    ) -> str:
        """
        Export summaries for a list of papers
        
        Args:
            papers: List of papers to summarize
            export_format: Export format
            summary_method: Summarization method
            include_key_findings: Whether to include key findings
            max_length: Maximum summary length
            
        Returns:
            Exported summaries as string
        """
        try:
            logger.info(f"Exporting summaries for {len(papers)} papers")
            
            summaries_data = []
            
            for paper in papers:
                # Generate summary
                text = paper.abstract or "No abstract available"
                
                summary_result = self.summarizer.summarize_paper(
                    text, summary_method, max_length
                )
                
                summary_entry = {
                    "paper_id": paper.id,
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "original_abstract": paper.abstract,
                    "summary": summary_result.summary,
                    "summary_method": summary_result.method,
                    "compression_ratio": summary_result.compression_ratio,
                    "confidence_score": summary_result.confidence_score,
                    "key_points": summary_result.key_points if include_key_findings else []
                }
                
                summaries_data.append(summary_entry)
            
            # Export based on format
            if export_format == ExportFormat.JSON:
                return self._export_json(summaries_data)
            elif export_format == ExportFormat.HTML:
                return self._export_summaries_html(summaries_data)
            elif export_format == ExportFormat.MARKDOWN:
                return self._export_summaries_markdown(summaries_data)
            elif export_format == ExportFormat.TXT:
                return self._export_summaries_txt(summaries_data)
            else:
                return self._export_json(summaries_data)
                
        except Exception as e:
            logger.error(f"Error exporting summaries: {str(e)}")
            raise
    
    async def export_search_results(
        self,
        search_query: str,
        search_results: List[Paper],
        export_format: str = ExportFormat.HTML,
        include_metadata: bool = True
    ) -> str:
        """
        Export search results
        
        Args:
            search_query: Original search query
            search_results: List of search result papers
            export_format: Export format
            include_metadata: Whether to include metadata
            
        Returns:
            Exported search results as string
        """
        try:
            logger.info(f"Exporting search results for query: {search_query}")
            
            search_data = {
                "query": search_query,
                "generated_at": datetime.now().isoformat(),
                "total_results": len(search_results),
                "results": []
            }
            
            for paper in search_results:
                result_entry = {
                    "paper_id": paper.id,
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.abstract,
                    "relevance_score": paper.relevance_score,
                    "url": paper.url,
                    "source": paper.source.value if paper.source else None
                }
                
                if include_metadata:
                    result_entry.update({
                        "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                        "journal": getattr(paper, 'journal', None),
                        "doi": getattr(paper, 'doi', None),
                        "categories": getattr(paper, 'categories', []),
                        "keywords": getattr(paper, 'keywords', [])
                    })
                
                search_data["results"].append(result_entry)
            
            # Export based on format
            if export_format == ExportFormat.JSON:
                return self._export_json(search_data)
            elif export_format == ExportFormat.HTML:
                return self._export_search_results_html(search_data)
            elif export_format == ExportFormat.MARKDOWN:
                return self._export_search_results_markdown(search_data)
            elif export_format == ExportFormat.CSV:
                return self._export_csv(search_data["results"])
            else:
                return self._export_json(search_data)
                
        except Exception as e:
            logger.error(f"Error exporting search results: {str(e)}")
            raise
    
    async def _prepare_export_data(
        self,
        papers: List[Paper],
        include_abstracts: bool,
        include_citations: bool,
        citation_style: str,
        include_summaries: bool,
        custom_fields: List[str]
    ) -> List[Dict[str, Any]]:
        """Prepare paper data for export"""
        export_data = []
        
        for paper in papers:
            paper_data = {
                "id": paper.id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
                "url": paper.url,
                "source": paper.source.value if paper.source else None
            }
            
            # Add abstract if requested
            if include_abstracts:
                paper_data["abstract"] = paper.abstract
            
            # Add citation if requested
            if include_citations:
                citation_data = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "published": paper.publication_date.isoformat() if paper.publication_date else None,
                    "journal": getattr(paper, 'journal', None),
                    "url": paper.url,
                    "doi": getattr(paper, 'doi', None),
                    "abstract": paper.abstract
                }
                
                citation_result = self.citation_service.generate_citation_from_paper(
                    citation_data, citation_style
                )
                paper_data["citation"] = citation_result["citation"]
            
            # Add summary if requested
            if include_summaries and paper.abstract:
                summary_result = self.summarizer.summarize_paper(
                    paper.abstract, "abstractive", 300
                )
                paper_data["summary"] = summary_result.summary
                paper_data["key_points"] = summary_result.key_points
            
            # Add custom fields if requested
            if custom_fields:
                for field in custom_fields:
                    if hasattr(paper, field):
                        paper_data[field] = getattr(paper, field)
            
            export_data.append(paper_data)
        
        return export_data
    
    def _export_json(self, data: Any) -> str:
        """Export data as JSON"""
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
    
    def _export_csv(self, data: List[Dict[str, Any]]) -> str:
        """Export data as CSV"""
        if not data:
            return ""
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()
    
    def _export_xml(self, data: List[Dict[str, Any]]) -> str:
        """Export data as XML"""
        root = ET.Element("papers")
        
        for paper_data in data:
            paper_elem = ET.SubElement(root, "paper")
            
            for key, value in paper_data.items():
                if isinstance(value, list):
                    list_elem = ET.SubElement(paper_elem, key)
                    for item in value:
                        item_elem = ET.SubElement(list_elem, "item")
                        item_elem.text = str(item)
                else:
                    elem = ET.SubElement(paper_elem, key)
                    elem.text = str(value) if value is not None else ""
        
        return ET.tostring(root, encoding='unicode', method='xml')
    
    def _export_txt(self, data: List[Dict[str, Any]]) -> str:
        """Export data as plain text"""
        output = []
        
        for i, paper_data in enumerate(data, 1):
            output.append(f"Paper {i}:")
            output.append(f"Title: {paper_data.get('title', 'N/A')}")
            output.append(f"Authors: {', '.join(paper_data.get('authors', []))}")
            output.append(f"Publication Date: {paper_data.get('publication_date', 'N/A')}")
            output.append(f"URL: {paper_data.get('url', 'N/A')}")
            
            if 'abstract' in paper_data:
                output.append(f"Abstract: {paper_data['abstract']}")
            
            if 'citation' in paper_data:
                output.append(f"Citation: {paper_data['citation']}")
            
            output.append("-" * 80)
        
        return "\n".join(output)
    
    def _export_html(self, data: List[Dict[str, Any]]) -> str:
        """Export data as HTML"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Academic Papers Export</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".paper { margin-bottom: 30px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }",
            ".title { font-size: 18px; font-weight: bold; color: #333; }",
            ".authors { color: #666; margin: 5px 0; }",
            ".abstract { margin: 10px 0; text-align: justify; }",
            ".citation { background-color: #f5f5f5; padding: 10px; margin: 10px 0; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Academic Papers Export</h1>",
            f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            f"<p>Total papers: {len(data)}</p>"
        ]
        
        for i, paper_data in enumerate(data, 1):
            html_parts.extend([
                f"<div class='paper'>",
                f"<h2>Paper {i}</h2>",
                f"<div class='title'>{paper_data.get('title', 'N/A')}</div>",
                f"<div class='authors'>Authors: {', '.join(paper_data.get('authors', []))}</div>",
                f"<div>Publication Date: {paper_data.get('publication_date', 'N/A')}</div>",
                f"<div>Source: {paper_data.get('source', 'N/A')}</div>"
            ])
            
            if paper_data.get('url'):
                html_parts.append(f"<div>URL: <a href='{paper_data['url']}'>{paper_data['url']}</a></div>")
            
            if 'abstract' in paper_data:
                html_parts.append(f"<div class='abstract'><strong>Abstract:</strong> {paper_data['abstract']}</div>")
            
            if 'citation' in paper_data:
                html_parts.append(f"<div class='citation'><strong>Citation:</strong> {paper_data['citation']}</div>")
            
            html_parts.append("</div>")
        
        html_parts.extend(["</body>", "</html>"])
        
        return "\n".join(html_parts)
    
    def _export_markdown(self, data: List[Dict[str, Any]]) -> str:
        """Export data as Markdown"""
        markdown_parts = [
            "# Academic Papers Export",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total papers: {len(data)}",
            ""
        ]
        
        for i, paper_data in enumerate(data, 1):
            markdown_parts.extend([
                f"## Paper {i}",
                f"**Title:** {paper_data.get('title', 'N/A')}",
                f"**Authors:** {', '.join(paper_data.get('authors', []))}",
                f"**Publication Date:** {paper_data.get('publication_date', 'N/A')}",
                f"**Source:** {paper_data.get('source', 'N/A')}"
            ])
            
            if paper_data.get('url'):
                markdown_parts.append(f"**URL:** [{paper_data['url']}]({paper_data['url']})")
            
            if 'abstract' in paper_data:
                markdown_parts.append(f"**Abstract:** {paper_data['abstract']}")
            
            if 'citation' in paper_data:
                markdown_parts.append(f"**Citation:** {paper_data['citation']}")
            
            markdown_parts.append("---")
        
        return "\n\n".join(markdown_parts)
    
    def _export_bibtex(self, papers: List[Paper]) -> str:
        """Export papers as BibTeX"""
        bibtex_entries = []
        
        for paper in papers:
            paper_data = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": paper.publication_date.isoformat() if paper.publication_date else None,
                "journal": getattr(paper, 'journal', None),
                "url": paper.url,
                "doi": getattr(paper, 'doi', None),
                "abstract": paper.abstract
            }
            
            citation_result = self.citation_service.generate_citation_from_paper(
                paper_data, "bibtex"
            )
            bibtex_entries.append(citation_result["citation"])
        
        return "\n\n".join(bibtex_entries)
    
    def _export_endnote(self, papers: List[Paper]) -> str:
        """Export papers as EndNote format"""
        endnote_entries = []
        
        for paper in papers:
            paper_data = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "published": paper.publication_date.isoformat() if paper.publication_date else None,
                "journal": getattr(paper, 'journal', None),
                "url": paper.url,
                "doi": getattr(paper, 'doi', None),
                "abstract": paper.abstract
            }
            
            citation_result = self.citation_service.generate_citation_from_paper(
                paper_data, "endnote"
            )
            endnote_entries.append(citation_result["citation"])
        
        return "\n\n".join(endnote_entries)
    
    def _export_ris(self, papers: List[Paper]) -> str:
        """Export papers as RIS format"""
        ris_entries = []
        
        for paper in papers:
            ris_entry = [
                "TY  - JOUR",  # Journal article
                f"TI  - {paper.title}",
                f"AB  - {paper.abstract or ''}",
                f"UR  - {paper.url or ''}",
                f"PY  - {paper.publication_date.year if paper.publication_date else ''}",
                f"JO  - {getattr(paper, 'journal', '') or ''}",
                f"DO  - {getattr(paper, 'doi', '') or ''}"
            ]
            
            # Add authors
            for author in paper.authors:
                ris_entry.append(f"AU  - {author.name}")
            
            ris_entry.append("ER  - ")
            
            ris_entries.append("\n".join(ris_entry))
        
        return "\n\n".join(ris_entries)
    
    def _export_citations_txt(self, citations: List[Dict[str, Any]]) -> str:
        """Export citations as plain text"""
        output = []
        
        for i, citation in enumerate(citations, 1):
            output.append(f"{i}. {citation['citation']}")
        
        return "\n\n".join(output)
    
    def _export_citations_html(self, citations: List[Dict[str, Any]]) -> str:
        """Export citations as HTML"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Bibliography</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".citation { margin-bottom: 20px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Bibliography</h1>",
            f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        
        for i, citation in enumerate(citations, 1):
            html_parts.append(f"<div class='citation'>{i}. {citation['citation']}</div>")
        
        html_parts.extend(["</body>", "</html>"])
        
        return "\n".join(html_parts)
    
    def _export_citations_markdown(self, citations: List[Dict[str, Any]]) -> str:
        """Export citations as Markdown"""
        markdown_parts = [
            "# Bibliography",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        for i, citation in enumerate(citations, 1):
            markdown_parts.append(f"{i}. {citation['citation']}")
        
        return "\n\n".join(markdown_parts)
    
    def _export_reading_list_html(self, data: Dict[str, Any]) -> str:
        """Export reading list as HTML"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Reading List</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".section { margin-bottom: 30px; }",
            ".stats { background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }",
            "ul { padding-left: 20px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Reading List</h1>",
            f"<p>Generated on: {data['generated_at']}</p>",
            "<div class='stats'>",
            f"<h2>Statistics</h2>",
            f"<p>Total papers read: {data['total_papers']}</p>",
            f"<p>Total bookmarks: {data['total_bookmarks']}</p>",
            "</div>"
        ]
        
        # Add sections
        sections = [
            ("Reading History", data['reading_history']),
            ("Bookmarks", data['bookmarks']),
            ("Favorite Authors", data['favorite_authors']),
            ("Favorite Journals", data['favorite_journals']),
            ("Favorite Categories", data['favorite_categories'])
        ]
        
        for section_name, section_data in sections:
            if section_data:
                html_parts.append(f"<div class='section'>")
                html_parts.append(f"<h2>{section_name}</h2>")
                html_parts.append("<ul>")
                for item in section_data:
                    html_parts.append(f"<li>{item}</li>")
                html_parts.append("</ul>")
                html_parts.append("</div>")
        
        html_parts.extend(["</body>", "</html>"])
        
        return "\n".join(html_parts)
    
    def _export_reading_list_markdown(self, data: Dict[str, Any]) -> str:
        """Export reading list as Markdown"""
        markdown_parts = [
            "# Reading List",
            f"Generated on: {data['generated_at']}",
            "",
            "## Statistics",
            f"- Total papers read: {data['total_papers']}",
            f"- Total bookmarks: {data['total_bookmarks']}",
            ""
        ]
        
        # Add sections
        sections = [
            ("Reading History", data['reading_history']),
            ("Bookmarks", data['bookmarks']),
            ("Favorite Authors", data['favorite_authors']),
            ("Favorite Journals", data['favorite_journals']),
            ("Favorite Categories", data['favorite_categories'])
        ]
        
        for section_name, section_data in sections:
            if section_data:
                markdown_parts.append(f"## {section_name}")
                for item in section_data:
                    markdown_parts.append(f"- {item}")
                markdown_parts.append("")
        
        return "\n".join(markdown_parts)
    
    def _export_reading_list_txt(self, data: Dict[str, Any]) -> str:
        """Export reading list as plain text"""
        output = [
            "READING LIST",
            "=" * 50,
            f"Generated on: {data['generated_at']}",
            "",
            "STATISTICS",
            "-" * 20,
            f"Total papers read: {data['total_papers']}",
            f"Total bookmarks: {data['total_bookmarks']}",
            ""
        ]
        
        # Add sections
        sections = [
            ("READING HISTORY", data['reading_history']),
            ("BOOKMARKS", data['bookmarks']),
            ("FAVORITE AUTHORS", data['favorite_authors']),
            ("FAVORITE JOURNALS", data['favorite_journals']),
            ("FAVORITE CATEGORIES", data['favorite_categories'])
        ]
        
        for section_name, section_data in sections:
            if section_data:
                output.append(section_name)
                output.append("-" * len(section_name))
                for item in section_data:
                    output.append(f"- {item}")
                output.append("")
        
        return "\n".join(output)
    
    def _export_summaries_html(self, data: List[Dict[str, Any]]) -> str:
        """Export summaries as HTML"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Paper Summaries</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".summary { margin-bottom: 30px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }",
            ".title { font-size: 18px; font-weight: bold; color: #333; }",
            ".authors { color: #666; margin: 5px 0; }",
            ".summary-text { margin: 10px 0; text-align: justify; }",
            ".key-points { background-color: #f9f9f9; padding: 10px; margin: 10px 0; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Paper Summaries</h1>",
            f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            f"<p>Total summaries: {len(data)}</p>"
        ]
        
        for i, summary_data in enumerate(data, 1):
            html_parts.extend([
                f"<div class='summary'>",
                f"<h2>Summary {i}</h2>",
                f"<div class='title'>{summary_data.get('title', 'N/A')}</div>",
                f"<div class='authors'>Authors: {', '.join(summary_data.get('authors', []))}</div>",
                f"<div class='summary-text'><strong>Summary:</strong> {summary_data.get('summary', 'N/A')}</div>",
                f"<div>Method: {summary_data.get('summary_method', 'N/A')}</div>",
                f"<div>Compression Ratio: {summary_data.get('compression_ratio', 'N/A')}</div>",
                f"<div>Confidence Score: {summary_data.get('confidence_score', 'N/A')}</div>"
            ])
            
            if summary_data.get('key_points'):
                html_parts.append("<div class='key-points'><strong>Key Points:</strong>")
                html_parts.append("<ul>")
                for point in summary_data['key_points']:
                    html_parts.append(f"<li>{point}</li>")
                html_parts.append("</ul></div>")
            
            html_parts.append("</div>")
        
        html_parts.extend(["</body>", "</html>"])
        
        return "\n".join(html_parts)
    
    def _export_summaries_markdown(self, data: List[Dict[str, Any]]) -> str:
        """Export summaries as Markdown"""
        markdown_parts = [
            "# Paper Summaries",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total summaries: {len(data)}",
            ""
        ]
        
        for i, summary_data in enumerate(data, 1):
            markdown_parts.extend([
                f"## Summary {i}",
                f"**Title:** {summary_data.get('title', 'N/A')}",
                f"**Authors:** {', '.join(summary_data.get('authors', []))}",
                f"**Summary:** {summary_data.get('summary', 'N/A')}",
                f"**Method:** {summary_data.get('summary_method', 'N/A')}",
                f"**Compression Ratio:** {summary_data.get('compression_ratio', 'N/A')}",
                f"**Confidence Score:** {summary_data.get('confidence_score', 'N/A')}"
            ])
            
            if summary_data.get('key_points'):
                markdown_parts.append("**Key Points:**")
                for point in summary_data['key_points']:
                    markdown_parts.append(f"- {point}")
            
            markdown_parts.append("---")
        
        return "\n\n".join(markdown_parts)
    
    def _export_summaries_txt(self, data: List[Dict[str, Any]]) -> str:
        """Export summaries as plain text"""
        output = [
            "PAPER SUMMARIES",
            "=" * 50,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total summaries: {len(data)}",
            ""
        ]
        
        for i, summary_data in enumerate(data, 1):
            output.extend([
                f"SUMMARY {i}",
                "-" * 20,
                f"Title: {summary_data.get('title', 'N/A')}",
                f"Authors: {', '.join(summary_data.get('authors', []))}",
                f"Summary: {summary_data.get('summary', 'N/A')}",
                f"Method: {summary_data.get('summary_method', 'N/A')}",
                f"Compression Ratio: {summary_data.get('compression_ratio', 'N/A')}",
                f"Confidence Score: {summary_data.get('confidence_score', 'N/A')}"
            ])
            
            if summary_data.get('key_points'):
                output.append("Key Points:")
                for point in summary_data['key_points']:
                    output.append(f"- {point}")
            
            output.append("")
        
        return "\n".join(output)
    
    def _export_search_results_html(self, data: Dict[str, Any]) -> str:
        """Export search results as HTML"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Search Results</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".result { margin-bottom: 30px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }",
            ".title { font-size: 18px; font-weight: bold; color: #333; }",
            ".authors { color: #666; margin: 5px 0; }",
            ".abstract { margin: 10px 0; text-align: justify; }",
            ".metadata { background-color: #f5f5f5; padding: 10px; margin: 10px 0; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Search Results</h1>",
            f"<p>Query: <strong>{data['query']}</strong></p>",
            f"<p>Generated on: {data['generated_at']}</p>",
            f"<p>Total results: {data['total_results']}</p>"
        ]
        
        for i, result in enumerate(data['results'], 1):
            html_parts.extend([
                f"<div class='result'>",
                f"<h2>Result {i}</h2>",
                f"<div class='title'>{result.get('title', 'N/A')}</div>",
                f"<div class='authors'>Authors: {', '.join(result.get('authors', []))}</div>",
                f"<div>Relevance Score: {result.get('relevance_score', 'N/A')}</div>",
                f"<div>Source: {result.get('source', 'N/A')}</div>"
            ])
            
            if result.get('url'):
                html_parts.append(f"<div>URL: <a href='{result['url']}'>{result['url']}</a></div>")
            
            if result.get('abstract'):
                html_parts.append(f"<div class='abstract'><strong>Abstract:</strong> {result['abstract']}</div>")
            
            html_parts.append("</div>")
        
        html_parts.extend(["</body>", "</html>"])
        
        return "\n".join(html_parts)
    
    def _export_search_results_markdown(self, data: Dict[str, Any]) -> str:
        """Export search results as Markdown"""
        markdown_parts = [
            "# Search Results",
            f"**Query:** {data['query']}",
            f"**Generated on:** {data['generated_at']}",
            f"**Total results:** {data['total_results']}",
            ""
        ]
        
        for i, result in enumerate(data['results'], 1):
            markdown_parts.extend([
                f"## Result {i}",
                f"**Title:** {result.get('title', 'N/A')}",
                f"**Authors:** {', '.join(result.get('authors', []))}",
                f"**Relevance Score:** {result.get('relevance_score', 'N/A')}",
                f"**Source:** {result.get('source', 'N/A')}"
            ])
            
            if result.get('url'):
                markdown_parts.append(f"**URL:** [{result['url']}]({result['url']})")
            
            if result.get('abstract'):
                markdown_parts.append(f"**Abstract:** {result['abstract']}")
            
            markdown_parts.append("---")
        
        return "\n\n".join(markdown_parts)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats"""
        return self.supported_formats
    
    def get_export_info(self) -> Dict[str, Any]:
        """Get information about export capabilities"""
        return {
            "supported_formats": self.supported_formats,
            "export_types": [
                "papers",
                "citations",
                "reading_list",
                "summaries",
                "search_results"
            ],
            "citation_styles": [style.value for style in CitationStyle],
            "features": {
                "include_abstracts": True,
                "include_citations": True,
                "include_summaries": True,
                "custom_fields": True,
                "multiple_formats": True
            }
        }

# Global instance
export_service = ExportService() 