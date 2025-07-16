"""Data sources package for Academic Research Assistant."""

from .arxiv_client import ArxivClient, arxiv_client
from .pubmed_client import PubMedClient, pubmed_client
from .scholar_client import GoogleScholarClient, google_scholar_client

__all__ = ["ArxivClient", "arxiv_client", "PubMedClient", "pubmed_client", "GoogleScholarClient", "google_scholar_client"] 