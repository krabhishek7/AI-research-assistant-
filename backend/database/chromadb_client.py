"""ChromaDB client for vector database operations."""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import uuid
import json

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from backend.database.models import Paper, PaperSource, SearchResult
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBClient:
    """Client for interacting with ChromaDB vector database."""
    
    def __init__(self):
        self.client = None
        self.collections = {}
        self.embedding_function = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collections."""
        try:
            # Create ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Set up embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            )
            
            # Initialize collections for different sources
            self._create_collections()
            
            logger.info("ChromaDB client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            raise
    
    def _create_collections(self):
        """Create collections for different paper sources."""
        collection_names = {
            PaperSource.ARXIV: "arxiv_papers",
            PaperSource.PUBMED: "pubmed_papers", 
            PaperSource.GOOGLE_SCHOLAR: "scholar_papers",
            PaperSource.MANUAL: "manual_papers"
        }
        
        for source, collection_name in collection_names.items():
            try:
                # Try to get existing collection or create new one
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"source": source.value}
                )
                self.collections[source] = collection
                logger.info(f"Collection '{collection_name}' ready")
                
            except Exception as e:
                logger.error(f"Error creating collection '{collection_name}': {str(e)}")
    
    def _paper_to_document(self, paper: Paper) -> Dict[str, Any]:
        """Convert Paper object to document format for ChromaDB."""
        # Create document text from title and abstract
        document_text = f"Title: {paper.title}\n\nAbstract: {paper.abstract}"
        
        # If full text is available, include it
        if paper.full_text:
            document_text += f"\n\nFull Text: {paper.full_text[:10000]}"  # Limit length
        
        # Create metadata
        metadata = {
            "paper_id": paper.id or str(uuid.uuid4()),
            "title": paper.title,
            "authors": ", ".join([author.name for author in paper.authors]),
            "source": paper.source.value,
            "categories": ", ".join(paper.categories) if paper.categories else "",
            "keywords": ", ".join(paper.keywords) if paper.keywords else "",
            "publication_date": paper.publication_date.isoformat() if paper.publication_date else "",
            "doi": paper.doi or "",
            "arxiv_id": paper.arxiv_id or "",
            "pubmed_id": paper.pubmed_id or "",
            "url": paper.url or "",
            "pdf_url": paper.pdf_url or "",
            "citation_count": paper.citation_count or 0,
            "created_at": paper.created_at.isoformat(),
            "updated_at": paper.updated_at.isoformat()
        }
        
        return {
            "document": document_text,
            "metadata": metadata,
            "id": metadata["paper_id"]
        }
    
    def _document_to_paper(self, document: Dict[str, Any]) -> Paper:
        """Convert ChromaDB document back to Paper object."""
        from backend.database.models import Author  # Import here to avoid circular imports
        
        metadata = document.get("metadata", {})
        
        # Parse authors
        authors = []
        if metadata.get("authors"):
            author_names = metadata["authors"].split(", ")
            authors = [Author(name=name.strip()) for name in author_names if name.strip()]
        
        # Parse categories and keywords
        categories = []
        if metadata.get("categories"):
            categories = [cat.strip() for cat in metadata["categories"].split(",") if cat.strip()]
        
        keywords = []
        if metadata.get("keywords"):
            keywords = [kw.strip() for kw in metadata["keywords"].split(",") if kw.strip()]
        
        # Parse dates
        publication_date = None
        if metadata.get("publication_date"):
            try:
                publication_date = datetime.fromisoformat(metadata["publication_date"])
            except:
                pass
        
        created_at = datetime.now()
        if metadata.get("created_at"):
            try:
                created_at = datetime.fromisoformat(metadata["created_at"])
            except:
                pass
        
        updated_at = datetime.now()
        if metadata.get("updated_at"):
            try:
                updated_at = datetime.fromisoformat(metadata["updated_at"])
            except:
                pass
        
        # Extract abstract from document text
        abstract = ""
        doc_text = document.get("document", "")
        if "Abstract:" in doc_text:
            abstract_start = doc_text.find("Abstract:") + 9
            abstract_end = doc_text.find("\n\nFull Text:")
            if abstract_end == -1:
                abstract_end = len(doc_text)
            abstract = doc_text[abstract_start:abstract_end].strip()
        
        return Paper(
            id=metadata.get("paper_id"),
            title=metadata.get("title", ""),
            authors=authors,
            abstract=abstract,
            publication_date=publication_date,
            doi=metadata.get("doi"),
            arxiv_id=metadata.get("arxiv_id"),
            pubmed_id=metadata.get("pubmed_id"),
            url=metadata.get("url"),
            pdf_url=metadata.get("pdf_url"),
            source=PaperSource(metadata.get("source", PaperSource.MANUAL.value)),
            categories=categories,
            keywords=keywords,
            citation_count=metadata.get("citation_count", 0),
            created_at=created_at,
            updated_at=updated_at,
            embedding_generated=True  # Since it's in ChromaDB
        )
    
    async def add_paper(self, paper: Paper) -> bool:
        """
        Add a paper to the appropriate collection.
        
        Args:
            paper: Paper object to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.collections.get(paper.source)
            if not collection:
                logger.error(f"No collection found for source: {paper.source}")
                return False
            
            # Convert paper to document format
            doc = self._paper_to_document(paper)
            
            # Add to collection
            collection.add(
                documents=[doc["document"]],
                metadatas=[doc["metadata"]],
                ids=[doc["id"]]
            )
            
            logger.info(f"Added paper '{paper.title}' to {paper.source.value} collection")
            return True
            
        except Exception as e:
            logger.error(f"Error adding paper to ChromaDB: {str(e)}")
            return False
    
    async def add_papers(self, papers: List[Paper]) -> int:
        """
        Add multiple papers to their appropriate collections.
        
        Args:
            papers: List of Paper objects to add
            
        Returns:
            Number of papers successfully added
        """
        added_count = 0
        
        # Group papers by source
        papers_by_source = {}
        for paper in papers:
            if paper.source not in papers_by_source:
                papers_by_source[paper.source] = []
            papers_by_source[paper.source].append(paper)
        
        # Add papers to their respective collections
        for source, source_papers in papers_by_source.items():
            try:
                collection = self.collections.get(source)
                if not collection:
                    logger.error(f"No collection found for source: {source}")
                    continue
                
                # Convert papers to document format
                documents = []
                metadatas = []
                ids = []
                
                for paper in source_papers:
                    doc = self._paper_to_document(paper)
                    documents.append(doc["document"])
                    metadatas.append(doc["metadata"])
                    ids.append(doc["id"])
                
                # Add to collection
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                added_count += len(source_papers)
                logger.info(f"Added {len(source_papers)} papers to {source.value} collection")
                
            except Exception as e:
                logger.error(f"Error adding papers to {source.value} collection: {str(e)}")
        
        return added_count
    
    async def search_papers(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[List[PaperSource]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Paper]:
        """
        Search papers using semantic similarity.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sources: List of sources to search (if None, search all)
            filters: Additional metadata filters
            
        Returns:
            List of Paper objects ordered by relevance
        """
        try:
            if sources is None:
                sources = list(self.collections.keys())
            
            all_results = []
            
            # Search in each specified collection
            for source in sources:
                collection = self.collections.get(source)
                if not collection:
                    continue
                
                try:
                    # Perform semantic search
                    query_params = {
                        "query_texts": [query],
                        "n_results": max_results
                    }
                    
                    # Only add where clause if filters are provided
                    if filters:
                        query_params["where"] = filters
                    
                    results = collection.query(**query_params)
                    
                    # Convert results to Paper objects
                    if results['documents'] and results['documents'][0]:
                        for i, doc in enumerate(results['documents'][0]):
                            document = {
                                "document": doc,
                                "metadata": results['metadatas'][0][i],
                                "distance": results['distances'][0][i]
                            }
                            
                            paper = self._document_to_paper(document)
                            # Add relevance score (convert distance to similarity)
                            paper.relevance_score = 1.0 - document["distance"]
                            all_results.append(paper)
                            
                except Exception as e:
                    logger.error(f"Error searching {source.value} collection: {str(e)}")
                    continue
            
            # Sort by relevance score and return top results
            all_results.sort(key=lambda x: x.relevance_score or 0, reverse=True)
            return all_results[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching papers: {str(e)}")
            return []
    
    async def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """
        Get a specific paper by its ID.
        
        Args:
            paper_id: Paper ID to search for
            
        Returns:
            Paper object if found, None otherwise
        """
        try:
            # Search all collections for the paper
            for source, collection in self.collections.items():
                try:
                    results = collection.get(
                        ids=[paper_id],
                        include=["documents", "metadatas"]
                    )
                    
                    if results['documents'] and results['documents'][0]:
                        document = {
                            "document": results['documents'][0],
                            "metadata": results['metadatas'][0]
                        }
                        return self._document_to_paper(document)
                        
                except Exception as e:
                    logger.error(f"Error searching for paper {paper_id} in {source.value}: {str(e)}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting paper by ID {paper_id}: {str(e)}")
            return None
    
    async def update_paper(self, paper: Paper) -> bool:
        """
        Update an existing paper in the database.
        
        Args:
            paper: Updated Paper object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.collections.get(paper.source)
            if not collection:
                logger.error(f"No collection found for source: {paper.source}")
                return False
            
            # Convert paper to document format
            doc = self._paper_to_document(paper)
            
            # Update the document
            collection.update(
                ids=[doc["id"]],
                documents=[doc["document"]],
                metadatas=[doc["metadata"]]
            )
            
            logger.info(f"Updated paper '{paper.title}' in {paper.source.value} collection")
            return True
            
        except Exception as e:
            logger.error(f"Error updating paper in ChromaDB: {str(e)}")
            return False
    
    async def delete_paper(self, paper_id: str) -> bool:
        """
        Delete a paper from the database.
        
        Args:
            paper_id: ID of paper to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Search all collections for the paper
            for source, collection in self.collections.items():
                try:
                    # Try to delete from this collection
                    collection.delete(ids=[paper_id])
                    logger.info(f"Deleted paper {paper_id} from {source.value} collection")
                    return True
                    
                except Exception as e:
                    # Paper might not exist in this collection, continue
                    continue
            
            logger.warning(f"Paper {paper_id} not found in any collection")
            return False
            
        except Exception as e:
            logger.error(f"Error deleting paper {paper_id}: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about all collections."""
        stats = {}
        
        for source, collection in self.collections.items():
            try:
                count = collection.count()
                stats[source.value] = {
                    "count": count,
                    "name": collection.name
                }
            except Exception as e:
                logger.error(f"Error getting stats for {source.value}: {str(e)}")
                stats[source.value] = {
                    "count": 0,
                    "error": str(e)
                }
        
        return stats
    
    def reset_collection(self, source: PaperSource) -> bool:
        """
        Reset (clear) a specific collection.
        
        Args:
            source: Paper source whose collection to reset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.collections.get(source)
            if not collection:
                logger.error(f"No collection found for source: {source}")
                return False
            
            # Delete the collection and recreate it
            self.client.delete_collection(collection.name)
            self._create_collections()
            
            logger.info(f"Reset collection for {source.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection for {source.value}: {str(e)}")
            return False

# Global instance
chromadb_client = ChromaDBClient() 