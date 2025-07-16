"""
Paper Summarization Service
Provides extractive and abstractive summarization using Hugging Face models
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    pipeline, BartForConditionalGeneration, BartTokenizer,
    T5ForConditionalGeneration, T5Tokenizer
)
from typing import Dict, List, Any, Optional, Union
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

@dataclass
class SummaryResult:
    """Data class for summary results"""
    summary: str
    method: str
    original_length: int
    summary_length: int
    compression_ratio: float
    key_points: List[str]
    confidence_score: float

class PaperSummarizer:
    """
    Comprehensive paper summarization service using multiple approaches
    """
    
    def __init__(self, 
                 model_name: str = "facebook/bart-large-cnn",
                 device: str = "auto",
                 max_length: int = 1024,
                 min_length: int = 50):
        """
        Initialize the summarizer
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run the model on ('auto', 'cpu', 'cuda')
            max_length: Maximum length of generated summary
            min_length: Minimum length of generated summary
        """
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing summarizer with model: {model_name} on device: {self.device}")
        
        # Initialize models and tokenizers
        self.tokenizer = None
        self.model = None
        self.summarizer_pipeline = None
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load the model
        self._load_model()
        
    def _load_model(self):
        """Load the summarization model"""
        try:
            # Load tokenizer and model based on model type
            if "bart" in self.model_name.lower():
                self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
                self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
            elif "t5" in self.model_name.lower():
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            else:
                # Generic approach
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            
            # Create pipeline
            self.summarizer_pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to a smaller model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if the primary model fails"""
        try:
            logger.info("Loading fallback model: facebook/bart-large-cnn")
            self.model_name = "facebook/bart-large-cnn"
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            self.summarizer_pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Fallback model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            self.model = None
            self.tokenizer = None
            self.summarizer_pipeline = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for summarization
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive periods
        text = re.sub(r'\.{3,}', '...', text)
        
        return text.strip()
    
    def extractive_summarization(self, text: str, num_sentences: int = 3) -> SummaryResult:
        """
        Perform extractive summarization using TF-IDF
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences to extract
            
        Returns:
            SummaryResult object
        """
        if not text:
            return SummaryResult("", "extractive", 0, 0, 0.0, [], 0.0)
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize into sentences
        sentences = sent_tokenize(processed_text)
        
        if len(sentences) <= num_sentences:
            return SummaryResult(
                summary=processed_text,
                method="extractive",
                original_length=len(text.split()),
                summary_length=len(processed_text.split()),
                compression_ratio=1.0,
                key_points=sentences,
                confidence_score=1.0
            )
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
        
        # Get top sentences
        top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_sentence_indices = sorted(top_sentence_indices)
        
        # Extract top sentences
        summary_sentences = [sentences[i] for i in top_sentence_indices]
        summary = ' '.join(summary_sentences)
        
        # Calculate key points
        key_points = self._extract_key_points(text)
        
        # Calculate confidence score
        confidence_score = np.mean(sentence_scores[top_sentence_indices]) / np.max(sentence_scores)
        
        return SummaryResult(
            summary=summary,
            method="extractive",
            original_length=len(text.split()),
            summary_length=len(summary.split()),
            compression_ratio=len(summary.split()) / len(text.split()),
            key_points=key_points,
            confidence_score=confidence_score
        )
    
    def abstractive_summarization(self, text: str, max_length: Optional[int] = None) -> SummaryResult:
        """
        Perform abstractive summarization using transformer models
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            
        Returns:
            SummaryResult object
        """
        if not text:
            return SummaryResult("", "abstractive", 0, 0, 0.0, [], 0.0)
        
        if not self.summarizer_pipeline:
            logger.error("No summarization model available")
            return SummaryResult("Model not available", "abstractive", 0, 0, 0.0, [], 0.0)
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Truncate text if too long
        max_input_length = 1024  # Common limit for many models
        if len(processed_text.split()) > max_input_length:
            processed_text = ' '.join(processed_text.split()[:max_input_length])
        
        max_len = max_length or self.max_length
        min_len = min(self.min_length, max_len // 2)
        
        try:
            # Generate summary
            if "t5" in self.model_name.lower():
                # T5 requires specific format
                input_text = f"summarize: {processed_text}"
            else:
                input_text = processed_text
            
            result = self.summarizer_pipeline(
                input_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True
            )
            
            summary = result[0]['summary_text']
            
            # Extract key points
            key_points = self._extract_key_points(text)
            
            # Calculate confidence score (simplified)
            confidence_score = min(1.0, len(summary.split()) / max(min_len, 1))
            
            return SummaryResult(
                summary=summary,
                method="abstractive",
                original_length=len(text.split()),
                summary_length=len(summary.split()),
                compression_ratio=len(summary.split()) / len(text.split()),
                key_points=key_points,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Abstractive summarization failed: {e}")
            # Fallback to extractive
            return self.extractive_summarization(text, 3)
    
    def _extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """
        Extract key points from text
        
        Args:
            text: Input text
            num_points: Number of key points to extract
            
        Returns:
            List of key points
        """
        if not text:
            return []
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_points:
            return sentences
        
        # Simple keyword extraction based on frequency
        words = word_tokenize(text.lower())
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word.isalpha() and word not in self.stop_words]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences based on word frequencies
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words 
                            if word.isalpha() and word not in self.stop_words]
            
            score = sum(word_freq.get(word, 0) for word in sentence_words)
            sentence_scores[sentence] = score
        
        # Get top sentences as key points
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        key_points = [sentence for sentence, score in top_sentences[:num_points]]
        
        return key_points
    
    def summarize_paper(self, paper_text: str, 
                       method: str = "abstractive",
                       max_length: Optional[int] = None) -> SummaryResult:
        """
        Summarize a research paper
        
        Args:
            paper_text: Full text of the paper
            method: Summarization method ('extractive' or 'abstractive')
            max_length: Maximum length of summary
            
        Returns:
            SummaryResult object
        """
        if not paper_text:
            return SummaryResult("No text provided", method, 0, 0, 0.0, [], 0.0)
        
        logger.info(f"Summarizing paper using {method} method")
        
        if method == "extractive":
            return self.extractive_summarization(paper_text, 5)
        elif method == "abstractive":
            return self.abstractive_summarization(paper_text, max_length)
        else:
            logger.error(f"Unknown summarization method: {method}")
            return self.abstractive_summarization(paper_text, max_length)
    
    def summarize_abstract(self, abstract: str, 
                          max_length: int = 100) -> SummaryResult:
        """
        Summarize a paper abstract (create a shorter version)
        
        Args:
            abstract: Paper abstract
            max_length: Maximum length of summary
            
        Returns:
            SummaryResult object
        """
        if not abstract:
            return SummaryResult("No abstract provided", "abstractive", 0, 0, 0.0, [], 0.0)
        
        # For abstracts, use extractive method with fewer sentences
        return self.extractive_summarization(abstract, 2)
    
    def extract_key_findings(self, paper_text: str, 
                           num_findings: int = 3) -> List[str]:
        """
        Extract key findings from a research paper
        
        Args:
            paper_text: Full text of the paper
            num_findings: Number of key findings to extract
            
        Returns:
            List of key findings
        """
        if not paper_text:
            return []
        
        # Look for sections that typically contain findings
        findings_keywords = [
            "results", "findings", "conclusion", "outcome", "discovery",
            "demonstrates", "shows", "indicates", "reveals", "suggests"
        ]
        
        sentences = sent_tokenize(paper_text)
        
        # Score sentences based on findings keywords
        finding_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for keyword in findings_keywords if keyword in sentence_lower)
            
            if score > 0:
                finding_sentences.append((sentence, score))
        
        # Sort by score and return top findings
        finding_sentences.sort(key=lambda x: x[1], reverse=True)
        
        return [sentence for sentence, score in finding_sentences[:num_findings]]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "pipeline_available": self.summarizer_pipeline is not None
        }

# Global summarizer instance
_summarizer = None

def get_summarizer() -> PaperSummarizer:
    """
    Get the global summarizer instance
    
    Returns:
        PaperSummarizer instance
    """
    global _summarizer
    if _summarizer is None:
        _summarizer = PaperSummarizer()
    return _summarizer

def summarize_paper_text(text: str, 
                        method: str = "abstractive",
                        max_length: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function to summarize paper text
    
    Args:
        text: Paper text to summarize
        method: Summarization method
        max_length: Maximum summary length
        
    Returns:
        Dictionary with summary results
    """
    summarizer = get_summarizer()
    result = summarizer.summarize_paper(text, method, max_length)
    
    return {
        "summary": result.summary,
        "method": result.method,
        "original_length": result.original_length,
        "summary_length": result.summary_length,
        "compression_ratio": result.compression_ratio,
        "key_points": result.key_points,
        "confidence_score": result.confidence_score
    } 