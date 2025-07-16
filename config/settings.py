import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    huggingface_token: Optional[str] = Field(None, env="HUGGINGFACE_TOKEN")
    pubmed_api_key: Optional[str] = Field(None, env="PUBMED_API_KEY")
    
    # Database Configuration
    chroma_persist_directory: str = Field("./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    
    # API Configuration
    arxiv_max_results: int = Field(100, env="ARXIV_MAX_RESULTS")
    pubmed_max_results: int = Field(100, env="PUBMED_MAX_RESULTS")
    
    # Rate Limiting
    arxiv_rate_limit: float = Field(3.0, env="ARXIV_RATE_LIMIT")  # seconds between requests
    pubmed_rate_limit: float = Field(0.34, env="PUBMED_RATE_LIMIT")  # 3 requests per second max
    
    # Model Configuration
    embedding_model: str = Field("allenai/specter", env="EMBEDDING_MODEL")
    summarization_model: str = Field("facebook/bart-large-cnn", env="SUMMARIZATION_MODEL")
    
    # Search Configuration
    default_search_results: int = Field(10, env="DEFAULT_SEARCH_RESULTS")
    max_search_results: int = Field(50, env="MAX_SEARCH_RESULTS")
    
    # Cache Configuration
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # 1 hour in seconds
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("app.log", env="LOG_FILE")
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Create necessary directories
Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True) 