"""
Recommendation Service
Provides personalized paper recommendations based on user behavior and preferences
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
from collections import defaultdict, Counter
import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.database.models import Paper, PaperSource, Author
from backend.database.chromadb_client import chromadb_client
from backend.services.search_service import search_service
from backend.processing.embeddings import embeddings_generator
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserProfile:
    """User profile for tracking preferences and behavior"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.reading_history: List[str] = []  # Paper IDs
        self.search_history: List[str] = []   # Search queries
        self.favorite_authors: List[str] = [] # Author names
        self.favorite_journals: List[str] = [] # Journal names
        self.favorite_categories: List[str] = [] # Categories/topics
        self.preferred_sources: List[PaperSource] = []
        self.reading_time: Dict[str, float] = {}  # Paper ID -> reading time
        self.ratings: Dict[str, int] = {}  # Paper ID -> rating (1-5)
        self.bookmarks: List[str] = []  # Paper IDs
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for storage"""
        return {
            "user_id": self.user_id,
            "reading_history": self.reading_history,
            "search_history": self.search_history,
            "favorite_authors": self.favorite_authors,
            "favorite_journals": self.favorite_journals,
            "favorite_categories": self.favorite_categories,
            "preferred_sources": [s.value for s in self.preferred_sources],
            "reading_time": self.reading_time,
            "ratings": self.ratings,
            "bookmarks": self.bookmarks,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create profile from dictionary"""
        profile = cls(data["user_id"])
        profile.reading_history = data.get("reading_history", [])
        profile.search_history = data.get("search_history", [])
        profile.favorite_authors = data.get("favorite_authors", [])
        profile.favorite_journals = data.get("favorite_journals", [])
        profile.favorite_categories = data.get("favorite_categories", [])
        profile.preferred_sources = [PaperSource(s) for s in data.get("preferred_sources", [])]
        profile.reading_time = data.get("reading_time", {})
        profile.ratings = data.get("ratings", {})
        profile.bookmarks = data.get("bookmarks", [])
        profile.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        profile.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        return profile

class RecommendationService:
    """
    Comprehensive recommendation service for academic papers
    """
    
    def __init__(self):
        self.chromadb_client = chromadb_client
        self.search_service = search_service
        self.embeddings_generator = embeddings_generator
        self.user_profiles: Dict[str, UserProfile] = {}
        self.default_user_id = "default"
        logger.info("Recommendation service initialized")
    
    def get_user_profile(self, user_id: str = None) -> UserProfile:
        """Get or create user profile"""
        if user_id is None:
            user_id = self.default_user_id
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)
        
        return self.user_profiles[user_id]
    
    def update_user_profile(self, user_id: str, profile: UserProfile):
        """Update user profile"""
        profile.updated_at = datetime.now()
        self.user_profiles[user_id] = profile
    
    async def get_recommendations(
        self,
        user_id: str = None,
        max_results: int = 10,
        recommendation_types: List[str] = None,
        exclude_read: bool = True
    ) -> Dict[str, List[Paper]]:
        """
        Get personalized recommendations for a user
        
        Args:
            user_id: User ID
            max_results: Maximum number of recommendations per type
            recommendation_types: Types of recommendations to generate
            exclude_read: Whether to exclude already read papers
            
        Returns:
            Dictionary of recommendation types and their papers
        """
        try:
            profile = self.get_user_profile(user_id)
            
            if recommendation_types is None:
                recommendation_types = [
                    "based_on_reading_history",
                    "similar_to_favorites",
                    "trending_papers",
                    "collaborative_filtering",
                    "content_based",
                    "diverse_exploration"
                ]
            
            recommendations = {}
            
            # Generate different types of recommendations
            for rec_type in recommendation_types:
                if rec_type == "based_on_reading_history":
                    papers = await self._recommend_based_on_reading_history(
                        profile, max_results, exclude_read
                    )
                elif rec_type == "similar_to_favorites":
                    papers = await self._recommend_similar_to_favorites(
                        profile, max_results, exclude_read
                    )
                elif rec_type == "trending_papers":
                    papers = await self._recommend_trending_papers(
                        profile, max_results, exclude_read
                    )
                elif rec_type == "collaborative_filtering":
                    papers = await self._recommend_collaborative_filtering(
                        profile, max_results, exclude_read
                    )
                elif rec_type == "content_based":
                    papers = await self._recommend_content_based(
                        profile, max_results, exclude_read
                    )
                elif rec_type == "diverse_exploration":
                    papers = await self._recommend_diverse_exploration(
                        profile, max_results, exclude_read
                    )
                else:
                    continue
                
                recommendations[rec_type] = papers
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return {}
    
    async def _recommend_based_on_reading_history(
        self,
        profile: UserProfile,
        max_results: int,
        exclude_read: bool
    ) -> List[Paper]:
        """Recommend papers based on reading history"""
        if not profile.reading_history:
            return []
        
        try:
            # Get recent papers from reading history
            recent_papers = profile.reading_history[-10:]  # Last 10 papers
            
            recommendations = []
            
            for paper_id in recent_papers:
                # Get the paper
                paper = await self.chromadb_client.get_paper_by_id(paper_id)
                if not paper:
                    continue
                
                # Find related papers
                related_papers = await self.search_service.find_related_papers(
                    paper=paper,
                    max_results=5,
                    sources=profile.preferred_sources or [PaperSource.ARXIV]
                )
                
                recommendations.extend(related_papers)
            
            # Remove duplicates and filter
            unique_recommendations = self._remove_duplicates(recommendations)
            
            if exclude_read:
                unique_recommendations = [
                    p for p in unique_recommendations 
                    if p.id not in profile.reading_history
                ]
            
            # Sort by relevance score
            unique_recommendations.sort(key=lambda p: p.relevance_score or 0, reverse=True)
            
            return unique_recommendations[:max_results]
            
        except Exception as e:
            logger.error(f"Error in reading history recommendations: {str(e)}")
            return []
    
    async def _recommend_similar_to_favorites(
        self,
        profile: UserProfile,
        max_results: int,
        exclude_read: bool
    ) -> List[Paper]:
        """Recommend papers similar to user's favorites"""
        if not profile.bookmarks:
            return []
        
        try:
            recommendations = []
            
            for paper_id in profile.bookmarks:
                # Get the favorite paper
                paper = await self.chromadb_client.get_paper_by_id(paper_id)
                if not paper:
                    continue
                
                # Find similar papers
                similar_papers = await self.search_service.find_related_papers(
                    paper=paper,
                    max_results=5,
                    sources=profile.preferred_sources or [PaperSource.ARXIV]
                )
                
                recommendations.extend(similar_papers)
            
            # Remove duplicates and filter
            unique_recommendations = self._remove_duplicates(recommendations)
            
            if exclude_read:
                unique_recommendations = [
                    p for p in unique_recommendations 
                    if p.id not in profile.reading_history
                ]
            
            # Sort by relevance score
            unique_recommendations.sort(key=lambda p: p.relevance_score or 0, reverse=True)
            
            return unique_recommendations[:max_results]
            
        except Exception as e:
            logger.error(f"Error in favorites recommendations: {str(e)}")
            return []
    
    async def _recommend_trending_papers(
        self,
        profile: UserProfile,
        max_results: int,
        exclude_read: bool
    ) -> List[Paper]:
        """Recommend trending/popular papers"""
        try:
            # Get recent papers from preferred categories
            recommendations = []
            
            for category in profile.favorite_categories:
                # Search for recent papers in this category
                category_papers = await self.search_service.search_papers(
                    query=category,
                    max_results=5,
                    sources=profile.preferred_sources or [PaperSource.ARXIV],
                    date_from=datetime.now() - timedelta(days=30)  # Last 30 days
                )
                
                recommendations.extend(category_papers.papers)
            
            # If no categories, get general trending papers
            if not recommendations and not profile.favorite_categories:
                trending_queries = [
                    "machine learning", "artificial intelligence", "deep learning",
                    "natural language processing", "computer vision", "data science"
                ]
                
                for query in trending_queries[:3]:  # Top 3 trending topics
                    trending_papers = await self.search_service.search_papers(
                        query=query,
                        max_results=3,
                        sources=profile.preferred_sources or [PaperSource.ARXIV],
                        date_from=datetime.now() - timedelta(days=7)  # Last week
                    )
                    
                    recommendations.extend(trending_papers.papers)
            
            # Remove duplicates and filter
            unique_recommendations = self._remove_duplicates(recommendations)
            
            if exclude_read:
                unique_recommendations = [
                    p for p in unique_recommendations 
                    if p.id not in profile.reading_history
                ]
            
            # Sort by publication date (newest first)
            unique_recommendations.sort(
                key=lambda p: p.publication_date or datetime.min, 
                reverse=True
            )
            
            return unique_recommendations[:max_results]
            
        except Exception as e:
            logger.error(f"Error in trending recommendations: {str(e)}")
            return []
    
    async def _recommend_collaborative_filtering(
        self,
        profile: UserProfile,
        max_results: int,
        exclude_read: bool
    ) -> List[Paper]:
        """Recommend papers based on collaborative filtering (simplified)"""
        try:
            # This is a simplified version - in a real system, you'd need
            # multiple users and their interactions
            
            recommendations = []
            
            # Find papers by favorite authors
            for author_name in profile.favorite_authors:
                author_papers = await self.search_service.search_papers(
                    query=f'author:"{author_name}"',
                    max_results=3,
                    sources=profile.preferred_sources or [PaperSource.ARXIV]
                )
                
                recommendations.extend(author_papers.papers)
            
            # Find papers from favorite journals
            for journal_name in profile.favorite_journals:
                journal_papers = await self.search_service.search_papers(
                    query=f'journal:"{journal_name}"',
                    max_results=3,
                    sources=profile.preferred_sources or [PaperSource.ARXIV]
                )
                
                recommendations.extend(journal_papers.papers)
            
            # Remove duplicates and filter
            unique_recommendations = self._remove_duplicates(recommendations)
            
            if exclude_read:
                unique_recommendations = [
                    p for p in unique_recommendations 
                    if p.id not in profile.reading_history
                ]
            
            return unique_recommendations[:max_results]
            
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {str(e)}")
            return []
    
    async def _recommend_content_based(
        self,
        profile: UserProfile,
        max_results: int,
        exclude_read: bool
    ) -> List[Paper]:
        """Recommend papers based on content similarity"""
        try:
            # Create a profile vector based on user's reading history
            if not profile.reading_history:
                return []
            
            # Get embeddings for read papers
            read_papers = []
            for paper_id in profile.reading_history[-20:]:  # Last 20 papers
                paper = await self.chromadb_client.get_paper_by_id(paper_id)
                if paper:
                    read_papers.append(paper)
            
            if not read_papers:
                return []
            
            # Generate embeddings for read papers
            paper_embeddings = await self.embeddings_generator.generate_paper_embeddings(
                read_papers
            )
            
            # Create user profile embedding (average of read papers)
            valid_embeddings = [emb for emb in paper_embeddings if emb is not None]
            if not valid_embeddings:
                return []
            
            user_profile_embedding = np.mean(valid_embeddings, axis=0)
            
            # Search for similar papers using the profile embedding
            similar_papers = await self.chromadb_client.search_by_embedding(
                embedding=user_profile_embedding.tolist(),
                max_results=max_results * 2,
                sources=profile.preferred_sources or [PaperSource.ARXIV]
            )
            
            # Remove duplicates and filter
            unique_recommendations = self._remove_duplicates(similar_papers)
            
            if exclude_read:
                unique_recommendations = [
                    p for p in unique_recommendations 
                    if p.id not in profile.reading_history
                ]
            
            return unique_recommendations[:max_results]
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {str(e)}")
            return []
    
    async def _recommend_diverse_exploration(
        self,
        profile: UserProfile,
        max_results: int,
        exclude_read: bool
    ) -> List[Paper]:
        """Recommend diverse papers for exploration"""
        try:
            # Get papers from different categories to encourage exploration
            exploration_queries = [
                "machine learning", "physics", "biology", "chemistry",
                "economics", "psychology", "neuroscience", "mathematics",
                "computer science", "engineering"
            ]
            
            recommendations = []
            
            # Get a few papers from each category
            for query in exploration_queries:
                if query not in profile.favorite_categories:  # Explore new areas
                    diverse_papers = await self.search_service.search_papers(
                        query=query,
                        max_results=2,
                        sources=profile.preferred_sources or [PaperSource.ARXIV]
                    )
                    
                    recommendations.extend(diverse_papers.papers)
            
            # Remove duplicates and filter
            unique_recommendations = self._remove_duplicates(recommendations)
            
            if exclude_read:
                unique_recommendations = [
                    p for p in unique_recommendations 
                    if p.id not in profile.reading_history
                ]
            
            # Shuffle for diversity
            import random
            random.shuffle(unique_recommendations)
            
            return unique_recommendations[:max_results]
            
        except Exception as e:
            logger.error(f"Error in diverse exploration: {str(e)}")
            return []
    
    def _remove_duplicates(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers from list"""
        seen = set()
        unique_papers = []
        
        for paper in papers:
            # Use ID if available, otherwise use title
            identifier = paper.id or paper.title
            if identifier not in seen:
                seen.add(identifier)
                unique_papers.append(paper)
        
        return unique_papers
    
    def track_user_interaction(
        self,
        user_id: str,
        interaction_type: str,
        paper_id: str,
        metadata: Dict[str, Any] = None
    ):
        """Track user interaction with a paper"""
        try:
            profile = self.get_user_profile(user_id)
            
            if interaction_type == "read":
                if paper_id not in profile.reading_history:
                    profile.reading_history.append(paper_id)
                
                # Track reading time if provided
                if metadata and "reading_time" in metadata:
                    profile.reading_time[paper_id] = metadata["reading_time"]
                    
            elif interaction_type == "bookmark":
                if paper_id not in profile.bookmarks:
                    profile.bookmarks.append(paper_id)
                    
            elif interaction_type == "rate":
                if metadata and "rating" in metadata:
                    profile.ratings[paper_id] = metadata["rating"]
                    
            elif interaction_type == "search":
                if metadata and "query" in metadata:
                    profile.search_history.append(metadata["query"])
            
            # Update preferences based on interaction
            self._update_preferences(profile, paper_id, interaction_type)
            
            # Update profile
            self.update_user_profile(user_id, profile)
            
        except Exception as e:
            logger.error(f"Error tracking interaction: {str(e)}")
    
    def _update_preferences(self, profile: UserProfile, paper_id: str, interaction_type: str):
        """Update user preferences based on interaction"""
        try:
            # This is a simplified version - in a real system, you'd analyze
            # the paper content to extract preferences
            
            # For now, we'll just track some basic patterns
            if interaction_type in ["read", "bookmark", "rate"]:
                # You could analyze the paper here and update:
                # - favorite_authors
                # - favorite_journals
                # - favorite_categories
                # - preferred_sources
                pass
                
        except Exception as e:
            logger.error(f"Error updating preferences: {str(e)}")
    
    def get_recommendation_explanation(
        self,
        paper: Paper,
        recommendation_type: str,
        user_profile: UserProfile
    ) -> str:
        """Get explanation for why a paper was recommended"""
        explanations = {
            "based_on_reading_history": f"Recommended because it's similar to papers you've read recently",
            "similar_to_favorites": f"Recommended because it's similar to your bookmarked papers",
            "trending_papers": f"Recommended because it's trending in your areas of interest",
            "collaborative_filtering": f"Recommended because other users with similar interests liked it",
            "content_based": f"Recommended because it matches your reading preferences",
            "diverse_exploration": f"Recommended to help you explore new research areas"
        }
        
        return explanations.get(recommendation_type, "Recommended for you")
    
    def get_user_stats(self, user_id: str = None) -> Dict[str, Any]:
        """Get user statistics"""
        try:
            profile = self.get_user_profile(user_id)
            
            return {
                "total_papers_read": len(profile.reading_history),
                "total_bookmarks": len(profile.bookmarks),
                "total_searches": len(profile.search_history),
                "favorite_authors": len(profile.favorite_authors),
                "favorite_journals": len(profile.favorite_journals),
                "favorite_categories": len(profile.favorite_categories),
                "average_rating": sum(profile.ratings.values()) / len(profile.ratings) if profile.ratings else 0,
                "total_reading_time": sum(profile.reading_time.values()) if profile.reading_time else 0,
                "account_age_days": (datetime.now() - profile.created_at).days,
                "most_read_categories": self._get_most_read_categories(profile),
                "reading_trends": self._get_reading_trends(profile)
            }
            
        except Exception as e:
            logger.error(f"Error getting user stats: {str(e)}")
            return {}
    
    def _get_most_read_categories(self, profile: UserProfile) -> List[str]:
        """Get most read categories based on reading history"""
        # This is a simplified implementation
        category_counts = Counter(profile.favorite_categories)
        return [category for category, count in category_counts.most_common(5)]
    
    def _get_reading_trends(self, profile: UserProfile) -> Dict[str, Any]:
        """Get reading trends over time"""
        # This is a simplified implementation
        return {
            "papers_this_week": len([p for p in profile.reading_history[-20:] if p]),
            "papers_this_month": len([p for p in profile.reading_history[-50:] if p]),
            "trend": "increasing" if len(profile.reading_history) > 10 else "stable"
        }

# Global instance
recommendation_service = RecommendationService() 