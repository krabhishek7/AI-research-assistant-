"""FastAPI main application for Academic Research Assistant."""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import time
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our services and models
from backend.database.models import (
    Paper, PaperSource, SearchResult, APIResponse, 
    SearchQuery, Author, CitationStyle
)
from backend.services.search_service import search_service
from backend.database.chromadb_client import chromadb_client
from backend.data_sources.arxiv_client import arxiv_client
from backend.data_sources.pubmed_client import pubmed_client
from backend.data_sources.scholar_client import google_scholar_client
from backend.processing.paper_processor import paper_processor
from backend.processing.summarizer import get_summarizer, summarize_paper_text
from backend.services.citation_service import get_citation_service, generate_citation_from_paper, CitationStyle
from backend.services.recommendation_service import recommendation_service
from backend.services.export_service import export_service
from config.settings import settings

# Add import for chat functionality
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Academic Research Assistant API",
    description="A comprehensive API for searching, analyzing, and managing academic papers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results")
    sources: List[Union[str, PaperSource]] = Field(default=[PaperSource.ARXIV], description="Sources to search")
    use_local_db: bool = Field(True, description="Search local database")
    use_external_apis: bool = Field(True, description="Search external APIs")
    date_from: Optional[datetime] = Field(None, description="Start date filter")
    date_to: Optional[datetime] = Field(None, description="End date filter")
    categories: List[str] = Field(default=[], description="Categories to filter by")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    context: Optional[str] = Field(None, description="Optional context from papers")
    max_length: int = Field(500, ge=50, le=2000, description="Maximum response length")
    temperature: float = Field(0.7, ge=0.1, le=1.0, description="Response creativity")

class RelatedPapersRequest(BaseModel):
    paper_id: str = Field(..., description="Paper ID to find related papers for")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results")
    sources: List[PaperSource] = Field(default=[PaperSource.ARXIV], description="Sources to search")

class SuggestionsRequest(BaseModel):
    partial_query: str = Field(..., description="Partial search query")

class PaperProcessRequest(BaseModel):
    paper_id: str = Field(..., description="Paper ID to process")

class ComparisonRequest(BaseModel):
    paper_ids: List[str] = Field(..., min_items=2, max_items=5, description="Paper IDs to compare")

class SummarizationRequest(BaseModel):
    paper_id: Optional[str] = Field(None, description="Paper ID to summarize")
    text: Optional[str] = Field(None, description="Text to summarize")
    method: str = Field("abstractive", description="Summarization method")
    max_length: int = Field(500, ge=50, le=2000, description="Maximum summary length")

class AbstractSummarizationRequest(BaseModel):
    abstract: str = Field(..., description="Abstract text to summarize")
    max_length: int = Field(200, ge=50, le=500, description="Maximum summary length")

class KeyFindingsRequest(BaseModel):
    paper_id: Optional[str] = Field(None, description="Paper ID to extract key findings from")
    text: Optional[str] = Field(None, description="Text to extract key findings from")
    max_findings: int = Field(5, ge=1, le=20, description="Maximum number of key findings")

class CitationRequest(BaseModel):
    paper_id: str = Field(..., description="Paper ID to generate citation for")
    style: CitationStyle = Field(CitationStyle.APA, description="Citation style")
    format_type: str = Field("text", description="Format type")

class BulkCitationRequest(BaseModel):
    paper_ids: List[str] = Field(..., description="List of paper IDs")
    style: CitationStyle = Field(CitationStyle.APA, description="Citation style")
    format_type: str = Field("text", description="Format type")

class CustomCitationRequest(BaseModel):
    paper_data: Dict[str, Any] = Field(..., description="Paper data for citation")
    style: CitationStyle = Field(CitationStyle.APA, description="Citation style")
    format_type: str = Field("text", description="Format type")

# API Endpoints

@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint with API information."""
    return APIResponse(
        success=True,
        message="Academic Research Assistant API",
        data={
            "version": "1.0.0",
            "description": "A comprehensive API for searching and analyzing academic papers",
            "endpoints": {
                "search": "POST /search",
                "get_paper": "GET /papers/{paper_id}",
                "related_papers": "POST /related",
                "suggestions": "POST /suggestions",
                "summarize": "POST /summarize",
                "summarize_abstract": "POST /summarize/abstract",
                "extract_key_findings": "POST /extract/key-findings",
                "summarizer_info": "GET /summarizer/info",
                "generate_citation": "POST /citations/generate",
                "bulk_citations": "POST /citations/bulk",
                "custom_citation": "POST /citations/custom",
                "citation_styles": "GET /citations/styles",
                "validate_citation": "POST /citations/validate",
                "recommendations": "POST /recommendations",
                "track_interaction": "POST /recommendations/track",
                "user_stats": "GET /recommendations/stats",
                "export_papers": "POST /export/papers",
                "export_citations": "POST /export/citations",
                "export_reading-list": "POST /export/reading-list",
                "export_summaries": "POST /export/summaries",
                "export_search-results": "POST /export/search-results",
                "export_info": "GET /export/info",
                "stats": "GET /stats",
                "health": "GET /health",
                "chat": "POST /chat"
            }
        }
    )

@app.post("/search", response_model=APIResponse)
async def search_papers(request: SearchRequest):
    """
    Search for academic papers across multiple sources.
    
    This endpoint provides comprehensive search functionality combining:
    - Local database (ChromaDB) with semantic search
    - External APIs (ArXiv, PubMed, Google Scholar)
    - Hybrid search (semantic + keyword)
    """
    try:
        logger.info(f"Search request: query='{request.query}', sources={request.sources}, use_local_db={request.use_local_db}")
        
        # Convert string sources to PaperSource enum if needed
        sources = []
        for source in request.sources:
            logger.info(f"Processing source: {source} (type: {type(source)})")
            if isinstance(source, str):
                if source == "arxiv":
                    sources.append(PaperSource.ARXIV)
                elif source == "pubmed":
                    sources.append(PaperSource.PUBMED)
                elif source == "google_scholar":
                    sources.append(PaperSource.GOOGLE_SCHOLAR)
                else:
                    logger.warning(f"Unknown source string: {source}")
            else:
                sources.append(source)
        
        logger.info(f"Converted sources: {sources}")
        
        # Perform search using search service
        results = await search_service.search_papers(
            query=request.query,
            max_results=request.max_results,
            sources=sources,
            use_local_db=request.use_local_db,
            use_external_apis=request.use_external_apis,
            date_from=request.date_from,
            date_to=request.date_to,
            categories=request.categories
        )
        
        # Debug: Log the results before returning
        logger.info(f"Search results: {len(results.papers)} papers")
        logger.info(f"Papers: {[p.title for p in results.papers]}")
        logger.info(f"Sources searched: {results.sources_searched}")
        
        # Create response data
        response_data = {
            "results": results.model_dump(),  # Use model_dump instead of deprecated dict()
            "query": request.query,
            "search_time": results.search_time,
            "sources_searched": [s.value for s in results.sources_searched]
        }
        
        # Debug: Log the response data
        logger.info(f"API Response data: results has {len(response_data['results']['papers'])} papers")
        
        return APIResponse(
            success=True,
            message=f"Found {results.total_results} papers",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/{paper_id:path}", response_model=APIResponse)
async def get_paper(paper_id: str = Path(..., description="Paper ID")):
    """Get detailed information about a specific paper."""
    try:
        # URL decode the paper ID
        from urllib.parse import unquote
        decoded_paper_id = unquote(paper_id)
        
        paper = await chromadb_client.get_paper_by_id(decoded_paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        return APIResponse(
            success=True,
            message="Paper retrieved successfully",
            data={"paper": paper.dict()}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving paper {paper_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/related", response_model=APIResponse)
async def find_related_papers(request: RelatedPapersRequest):
    """Find papers related to a given paper."""
    try:
        # Get the reference paper
        reference_paper = await chromadb_client.get_paper_by_id(request.paper_id)
        
        if not reference_paper:
            raise HTTPException(status_code=404, detail="Reference paper not found")
        
        # Find related papers
        related_papers = await search_service.find_related_papers(
            paper=reference_paper,
            max_results=request.max_results,
            sources=request.sources
        )
        
        return APIResponse(
            success=True,
            message=f"Found {len(related_papers)} related papers",
            data={
                "reference_paper": reference_paper.dict(),
                "related_papers": [p.dict() for p in related_papers]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding related papers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggestions", response_model=APIResponse)
async def get_search_suggestions(request: SuggestionsRequest):
    """Get search suggestions for a partial query."""
    try:
        suggestions = await search_service.get_search_suggestions(request.partial_query)
        
        return APIResponse(
            success=True,
            message=f"Found {len(suggestions)} suggestions",
            data={"suggestions": suggestions}
        )
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=APIResponse)
async def process_paper(request: PaperProcessRequest):
    """Process a paper to extract full text and additional information."""
    try:
        # Get the paper
        paper = await chromadb_client.get_paper_by_id(request.paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        # Process the paper
        processed_paper = await paper_processor.process_paper(paper)
        
        # Update in database
        await chromadb_client.update_paper(processed_paper)
        
        return APIResponse(
            success=True,
            message="Paper processed successfully",
            data={"paper": processed_paper.dict()}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing paper: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare", response_model=APIResponse)
async def compare_papers(request: ComparisonRequest):
    """Compare multiple papers."""
    try:
        # Get all papers
        papers = []
        for paper_id in request.paper_ids:
            paper = await chromadb_client.get_paper_by_id(paper_id)
            if paper:
                papers.append(paper)
        
        if len(papers) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 valid papers for comparison")
        
        # Create comparison data
        comparison_data = {
            "papers": [p.dict() for p in papers],
            "comparison": {
                "publication_years": [p.publication_date.year if p.publication_date else None for p in papers],
                "categories": [p.categories for p in papers],
                "authors": [len(p.authors) for p in papers],
                "common_keywords": _find_common_keywords(papers),
                "summary": _generate_comparison_summary(papers)
            }
        }
        
        return APIResponse(
            success=True,
            message=f"Compared {len(papers)} papers",
            data=comparison_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing papers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=APIResponse)
async def get_stats():
    """Get system statistics."""
    try:
        stats = search_service.get_search_stats()
        
        return APIResponse(
            success=True,
            message="Statistics retrieved successfully",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize", response_model=APIResponse)
async def summarize_paper(request: SummarizationRequest):
    """
    Summarize a research paper or provided text.
    
    This endpoint provides paper summarization using:
    - Abstractive summarization (AI-generated summaries)
    - Extractive summarization (key sentence extraction)
    - Key points extraction
    """
    try:
        logger.info(f"Summarization request: method={request.method}, paper_id={request.paper_id}")
        
        # Determine input text
        if request.paper_id:
            # Get paper from database
            paper = await chromadb_client.get_paper_by_id(request.paper_id)
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found")
            
            # Use abstract if available, otherwise use full text if available
            text = paper.abstract
            if not text and hasattr(paper, 'full_text'):
                text = paper.full_text
            
            if not text:
                raise HTTPException(status_code=400, detail="No text available for summarization")
                
        elif request.text:
            text = request.text
        else:
            raise HTTPException(status_code=400, detail="Either paper_id or text must be provided")
        
        # Perform summarization
        summarizer = get_summarizer()
        result = summarizer.summarize_paper(text, request.method, request.max_length)
        
        return APIResponse(
            success=True,
            message="Summarization completed successfully",
            data={
                "summary": result.summary,
                "method": result.method,
                "original_length": result.original_length,
                "summary_length": result.summary_length,
                "compression_ratio": result.compression_ratio,
                "key_points": result.key_points,
                "confidence_score": result.confidence_score,
                "summarizer_model": summarizer.model_name
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/abstract", response_model=APIResponse)
async def summarize_abstract(request: AbstractSummarizationRequest):
    """
    Summarize a paper abstract to create a shorter version.
    
    This is useful for creating concise summaries of already condensed abstracts.
    """
    try:
        logger.info(f"Abstract summarization request: length={len(request.abstract)}")
        
        # Perform abstract summarization
        summarizer = get_summarizer()
        result = summarizer.summarize_abstract(request.abstract, request.max_length)
        
        return APIResponse(
            success=True,
            message="Abstract summarization completed successfully",
            data={
                "summary": result.summary,
                "method": result.method,
                "original_length": result.original_length,
                "summary_length": result.summary_length,
                "compression_ratio": result.compression_ratio,
                "key_points": result.key_points,
                "confidence_score": result.confidence_score
            }
        )
        
    except Exception as e:
        logger.error(f"Error in abstract summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract/key-findings", response_model=APIResponse)
async def extract_key_findings(request: KeyFindingsRequest):
    """
    Extract key findings from a research paper.
    
    This endpoint identifies and extracts the most important findings,
    results, and conclusions from the paper.
    """
    try:
        logger.info(f"Key findings extraction request: paper_id={request.paper_id}")
        
        # Determine input text
        if request.paper_id:
            # Get paper from database
            paper = await chromadb_client.get_paper_by_id(request.paper_id)
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found")
            
            # Use full text if available, otherwise use abstract
            text = getattr(paper, 'full_text', None) or paper.abstract
            
            if not text:
                raise HTTPException(status_code=400, detail="No text available for key findings extraction")
                
        elif request.text:
            text = request.text
        else:
            raise HTTPException(status_code=400, detail="Either paper_id or text must be provided")
        
        # Extract key findings
        summarizer = get_summarizer()
        key_findings = summarizer.extract_key_findings(text, request.max_findings)
        
        return APIResponse(
            success=True,
            message="Key findings extracted successfully",
            data={
                "key_findings": key_findings,
                "total_findings": len(key_findings),
                "text_length": len(text.split())
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in key findings extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summarizer/info", response_model=APIResponse)
async def get_summarizer_info():
    """Get information about the summarization model."""
    try:
        summarizer = get_summarizer()
        model_info = summarizer.get_model_info()
        
        return APIResponse(
            success=True,
            message="Summarizer information retrieved successfully",
            data=model_info
        )
        
    except Exception as e:
        logger.error(f"Error getting summarizer info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/citations/generate", response_model=APIResponse)
async def generate_citation(request: CitationRequest):
    """
    Generate a citation for a specific paper.
    
    This endpoint creates properly formatted citations in various academic styles
    including APA, MLA, Chicago, Harvard, IEEE, Vancouver, and BibTeX.
    """
    try:
        logger.info(f"Citation generation request: paper_id={request.paper_id}, style={request.style}")
        
        # Get paper from database
        paper = await chromadb_client.get_paper_by_id(request.paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        # Convert paper to dict format
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
        
        # Generate citation
        citation_result = generate_citation_from_paper(paper_data, request.style)
        
        return APIResponse(
            success=True,
            message="Citation generated successfully",
            data={
                "citation": citation_result["citation"],
                "style": citation_result["style"],
                "format_type": request.format_type,
                "validation": citation_result["validation"],
                "paper_type": citation_result["paper_type"],
                "authors_count": citation_result["authors_count"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating citation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/citations/bulk", response_model=APIResponse)
async def generate_bulk_citations(request: BulkCitationRequest):
    """
    Generate citations for multiple papers.
    
    This endpoint allows bulk generation of citations for multiple papers
    in the same style and format.
    """
    try:
        logger.info(f"Bulk citation generation request: {len(request.paper_ids)} papers, style={request.style}")
        
        citations = []
        errors = []
        
        for paper_id in request.paper_ids:
            try:
                # Get paper from database
                paper = await chromadb_client.get_paper_by_id(paper_id)
                if not paper:
                    errors.append(f"Paper {paper_id} not found")
                    continue
                
                # Convert paper to dict format
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
                
                # Generate citation
                citation_result = generate_citation_from_paper(paper_data, request.style)
                citations.append({
                    "paper_id": paper_id,
                    "citation": citation_result["citation"],
                    "validation": citation_result["validation"]
                })
                
            except Exception as e:
                errors.append(f"Error generating citation for {paper_id}: {str(e)}")
        
        # Format citations for export
        citation_service = get_citation_service()
        
        if request.format_type == "text":
            formatted_citations = []
            for i, citation_data in enumerate(citations, 1):
                formatted_citations.append(f"{i}. {citation_data['citation']}")
            formatted_output = "\n\n".join(formatted_citations)
        else:
            formatted_output = "\n\n".join([c["citation"] for c in citations])
        
        return APIResponse(
            success=True,
            message=f"Generated {len(citations)} citations successfully",
            data={
                "citations": citations,
                "formatted_output": formatted_output,
                "style": request.style,
                "format_type": request.format_type,
                "total_papers": len(request.paper_ids),
                "successful_citations": len(citations),
                "errors": errors
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating bulk citations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/citations/custom", response_model=APIResponse)
async def generate_custom_citation(request: CustomCitationRequest):
    """
    Generate a citation from custom paper data.
    
    This endpoint allows generation of citations from custom paper metadata
    without requiring the paper to be in the database.
    """
    try:
        logger.info(f"Custom citation generation request: style={request.style}")
        
        # Generate citation from custom data
        citation_result = generate_citation_from_paper(request.paper_data, request.style)
        
        return APIResponse(
            success=True,
            message="Custom citation generated successfully",
            data={
                "citation": citation_result["citation"],
                "style": citation_result["style"],
                "format_type": request.format_type,
                "validation": citation_result["validation"],
                "paper_type": citation_result["paper_type"],
                "authors_count": citation_result["authors_count"]
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating custom citation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/citations/styles", response_model=APIResponse)
async def get_citation_styles():
    """Get list of supported citation styles."""
    try:
        citation_service = get_citation_service()
        styles = citation_service.get_supported_styles()
        
        return APIResponse(
            success=True,
            message="Citation styles retrieved successfully",
            data={
                "styles": styles,
                "total_styles": len(styles),
                "default_style": "apa"
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting citation styles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/citations/validate", response_model=APIResponse)
async def validate_citation_data(request: CustomCitationRequest):
    """
    Validate paper data for citation generation.
    
    This endpoint validates paper metadata and provides quality scores
    and suggestions for improvement.
    """
    try:
        logger.info("Citation validation request")
        
        # Create citation service
        citation_service = get_citation_service()
        
        # Create citation object
        citation = citation_service.create_citation(request.paper_data)
        
        # Validate citation
        validation = citation_service.validate_citation(citation)
        
        return APIResponse(
            success=True,
            message="Citation validation completed",
            data={
                "validation": validation,
                "paper_type": citation.paper_type.value,
                "authors_count": len(citation.authors),
                "has_year": citation.year is not None,
                "has_doi": citation.doi is not None,
                "has_journal": citation.journal is not None
            }
        )
        
    except Exception as e:
        logger.error(f"Error validating citation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if services are working
        health_status = {
            "api": "healthy",
            "database": "healthy",
            "search_service": "healthy",
            "arxiv_client": "healthy",
            "summarizer": "healthy",
            "citation_service": "healthy",
            "timestamp": datetime.now().isoformat()
        }
        
        # Test database connection
        try:
            stats = chromadb_client.get_collection_stats()
            health_status["database_stats"] = stats
        except Exception as e:
            health_status["database"] = f"error: {str(e)}"
        
        # Test summarizer
        try:
            summarizer = get_summarizer()
            summarizer_info = summarizer.get_model_info()
            health_status["summarizer_info"] = summarizer_info
        except Exception as e:
            health_status["summarizer"] = f"error: {str(e)}"
        
        # Test citation service
        try:
            citation_service = get_citation_service()
            citation_styles = citation_service.get_supported_styles()
            health_status["citation_service_info"] = {
                "supported_styles": citation_styles,
                "total_styles": len(citation_styles)
            }
        except Exception as e:
            health_status["citation_service"] = f"error: {str(e)}"
        
        return APIResponse(
            success=True,
            message="System is healthy",
            data=health_status
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Recommendation endpoints
@app.post("/recommendations", response_model=APIResponse)
async def get_recommendations(user_id: str = "default", max_results: int = 10):
    """Get personalized recommendations for a user."""
    try:
        recommendations = await recommendation_service.get_recommendations(
            user_id=user_id,
            max_results=max_results
        )
        
        # Convert papers to dict format for response
        recommendations_dict = {}
        for rec_type, papers in recommendations.items():
            recommendations_dict[rec_type] = [paper.dict() for paper in papers]
        
        return APIResponse(
            success=True,
            message="Recommendations retrieved successfully",
            data=recommendations_dict
        )
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations/track", response_model=APIResponse)
async def track_user_interaction(
    user_id: str,
    interaction_type: str,
    paper_id: str,
    metadata: Dict[str, Any] = None
):
    """Track user interaction with a paper."""
    try:
        recommendation_service.track_user_interaction(
            user_id=user_id,
            interaction_type=interaction_type,
            paper_id=paper_id,
            metadata=metadata or {}
        )
        
        return APIResponse(
            success=True,
            message="User interaction tracked successfully",
            data={}
        )
        
    except Exception as e:
        logger.error(f"Error tracking user interaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations/stats", response_model=APIResponse)
async def get_user_stats(user_id: str = "default"):
    """Get user statistics."""
    try:
        stats = recommendation_service.get_user_stats(user_id)
        
        return APIResponse(
            success=True,
            message="User statistics retrieved successfully",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Error getting user stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Export endpoints
@app.post("/export/papers", response_model=APIResponse)
async def export_papers(
    paper_ids: List[str],
    export_format: str = "json",
    include_abstracts: bool = True,
    include_citations: bool = True,
    citation_style: str = "apa",
    include_summaries: bool = False
):
    """Export papers in various formats."""
    try:
        # Get papers from database
        papers = []
        for paper_id in paper_ids:
            paper = await chromadb_client.get_paper_by_id(paper_id)
            if paper:
                papers.append(paper)
        
        if not papers:
            raise HTTPException(status_code=404, detail="No papers found")
        
        # Export papers
        exported_data = await export_service.export_papers(
            papers=papers,
            export_format=export_format,
            include_abstracts=include_abstracts,
            include_citations=include_citations,
            citation_style=citation_style,
            include_summaries=include_summaries
        )
        
        return APIResponse(
            success=True,
            message="Papers exported successfully",
            data={
                "exported_data": exported_data,
                "format": export_format,
                "paper_count": len(papers)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting papers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/citations", response_model=APIResponse)
async def export_citations(
    paper_ids: List[str],
    citation_style: str = "apa",
    export_format: str = "txt",
    include_metadata: bool = True
):
    """Export citations for papers."""
    try:
        # Get papers from database
        papers = []
        for paper_id in paper_ids:
            paper = await chromadb_client.get_paper_by_id(paper_id)
            if paper:
                papers.append(paper)
        
        if not papers:
            raise HTTPException(status_code=404, detail="No papers found")
        
        # Export citations
        exported_citations = await export_service.export_citations(
            papers=papers,
            citation_style=citation_style,
            export_format=export_format,
            include_metadata=include_metadata
        )
        
        return APIResponse(
            success=True,
            message="Citations exported successfully",
            data={
                "exported_data": exported_citations,
                "format": export_format,
                "style": citation_style,
                "paper_count": len(papers)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting citations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/reading-list", response_model=APIResponse)
async def export_reading_list(
    user_id: str = "default",
    export_format: str = "html",
    include_summaries: bool = True,
    include_notes: bool = True,
    include_ratings: bool = True
):
    """Export user's reading list."""
    try:
        exported_data = await export_service.export_reading_list(
            user_id=user_id,
            export_format=export_format,
            include_summaries=include_summaries,
            include_notes=include_notes,
            include_ratings=include_ratings
        )
        
        return APIResponse(
            success=True,
            message="Reading list exported successfully",
            data={
                "exported_data": exported_data,
                "format": export_format,
                "user_id": user_id
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting reading list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/summaries", response_model=APIResponse)
async def export_summaries(
    paper_ids: List[str],
    export_format: str = "html",
    summary_method: str = "abstractive",
    include_key_findings: bool = True,
    max_length: int = 500
):
    """Export summaries for papers."""
    try:
        # Get papers from database
        papers = []
        for paper_id in paper_ids:
            paper = await chromadb_client.get_paper_by_id(paper_id)
            if paper:
                papers.append(paper)
        
        if not papers:
            raise HTTPException(status_code=404, detail="No papers found")
        
        # Export summaries
        exported_summaries = await export_service.export_summaries(
            papers=papers,
            export_format=export_format,
            summary_method=summary_method,
            include_key_findings=include_key_findings,
            max_length=max_length
        )
        
        return APIResponse(
            success=True,
            message="Summaries exported successfully",
            data={
                "exported_data": exported_summaries,
                "format": export_format,
                "method": summary_method,
                "paper_count": len(papers)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting summaries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/search-results", response_model=APIResponse)
async def export_search_results(
    search_query: str,
    paper_ids: List[str],
    export_format: str = "html",
    include_metadata: bool = True
):
    """Export search results."""
    try:
        # Get papers from database
        papers = []
        for paper_id in paper_ids:
            paper = await chromadb_client.get_paper_by_id(paper_id)
            if paper:
                papers.append(paper)
        
        if not papers:
            raise HTTPException(status_code=404, detail="No papers found")
        
        # Export search results
        exported_data = await export_service.export_search_results(
            search_query=search_query,
            search_results=papers,
            export_format=export_format,
            include_metadata=include_metadata
        )
        
        return APIResponse(
            success=True,
            message="Search results exported successfully",
            data={
                "exported_data": exported_data,
                "format": export_format,
                "query": search_query,
                "paper_count": len(papers)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting search results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export/info", response_model=APIResponse)
async def get_export_info():
    """Get information about export capabilities."""
    try:
        export_info = export_service.get_export_info()
        
        return APIResponse(
            success=True,
            message="Export information retrieved successfully",
            data=export_info
        )
        
    except Exception as e:
        logger.error(f"Error getting export info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Global chat model instance (lazy loading)
_chat_model = None
_chat_tokenizer = None

def get_chat_model():
    """Get or initialize the chat model."""
    global _chat_model, _chat_tokenizer
    
    if _chat_model is None:
        try:
            # Use a free, lightweight conversational model
            model_name = "microsoft/DialoGPT-small"
            logger.info(f"Loading chat model: {model_name}")
            
            _chat_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _chat_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if not present
            if _chat_tokenizer.pad_token is None:
                _chat_tokenizer.pad_token = _chat_tokenizer.eos_token
            
            logger.info("Chat model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading chat model: {str(e)}")
            raise
    
    return _chat_model, _chat_tokenizer

# Chat endpoint
@app.post("/chat", response_model=APIResponse)
async def chat_with_ai(request: dict):
    """Chat with AI assistant."""
    try:
        message = request.get("message", "")
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["quantum", "physics"]):
            response = "Quantum physics deals with matter and energy at atomic/subatomic scales. Key principles include wave-particle duality, uncertainty, and entanglement. Applications include quantum computing and cryptography. What specific quantum concepts interest you?"
        elif any(word in message_lower for word in ["machine learning", "ai", "neural"]):
            response = "Machine learning enables computers to learn from data without explicit programming. Main types: supervised (labeled data), unsupervised (pattern finding), reinforcement (reward-based). Neural networks power deep learning. What ML aspect would you like to explore?"
        elif any(word in message_lower for word in ["research", "methodology"]):
            response = "Research methodology involves: 1) Clear research questions, 2) Appropriate methods (quantitative/qualitative), 3) Reliable data collection, 4) Systematic analysis, 5) Valid conclusions. What research area interests you?"
        else:
            response = f"I understand you are asking about \"{message}\". This is an interesting research topic! I can help you explore theoretical foundations, practical applications, recent developments, or research methodologies. What specific aspect would you like to focus on?"
        
        return APIResponse(
            success=True,
            message="Chat response generated",
            data={
                "response": response,
                "model": "Research Assistant",
                "message_length": len(message),
                "response_length": len(response)
            }
        )
    except Exception as e:
        return APIResponse(
            success=False,
            message="Chat error",
            data={
                "response": "I apologize, but I am having trouble processing your request. Please try rephrasing your question or use the search function to find relevant research papers.",
                "error": str(e)
            }
        )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )

