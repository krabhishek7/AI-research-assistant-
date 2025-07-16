# Academic Research Assistant

A comprehensive AI-powered research assistant that helps academics and researchers discover, analyze, and manage scientific literature from multiple sources including ArXiv, PubMed, and Google Scholar.

## Features

- **Multi-source Search**: Search across ArXiv, PubMed, and Google Scholar
- **Semantic Search**: AI-powered semantic search using scientific paper embeddings
- **Paper Summarization**: Automatic summarization of research papers
- **Citation Generation**: Generate citations in multiple formats (APA, MLA, Chicago, BibTeX)
- **Research Management**: Organize papers into research projects
- **Related Work Discovery**: Find related papers and recommendations
- **Export Functionality**: Export summaries, citations, and reading lists

## Project Structure

```
academic-research-assistant/
├── backend/
│   ├── data_sources/          # API clients for different sources
│   ├── processing/            # Text processing and AI models
│   ├── database/             # Vector database and data models
│   ├── services/             # Business logic services
│   └── api/                  # FastAPI endpoints
├── frontend/
│   └── app.py               # Gradio interface
├── config/
│   └── settings.py          # Configuration management
├── requirements.txt
├── .env.example
└── README.md
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd academic-research-assistant
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. **Install additional dependencies** (if needed):
```bash
# For PDF processing
pip install poppler-utils  # Linux/Mac
# For Windows, download poppler binaries
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

- `OPENAI_API_KEY`: OpenAI API key for advanced features
- `HUGGINGFACE_TOKEN`: Hugging Face token for model access
- `PUBMED_API_KEY`: PubMed API key (optional, for higher rate limits)
- `CHROMA_PERSIST_DIRECTORY`: Directory for vector database storage
- `ARXIV_MAX_RESULTS`: Maximum results from ArXiv API
- `EMBEDDING_MODEL`: Model for generating embeddings
- `SUMMARIZATION_MODEL`: Model for text summarization

### API Keys Setup

1. **Hugging Face**: Get your token from [Hugging Face](https://huggingface.co/settings/tokens)
2. **OpenAI**: Get your API key from [OpenAI](https://platform.openai.com/api-keys)
3. **PubMed**: Register at [NCBI](https://www.ncbi.nlm.nih.gov/books/NBK25497/)

## Usage

### Starting the Application

1. **Start the backend API**:
```bash
cd backend
python api/main.py
```

2. **Start the frontend interface**:
```bash
cd frontend
python app.py
```

3. **Access the application**:
   - API: http://localhost:8000
   - Frontend: http://localhost:7860

### API Endpoints

- `POST /search`: Search papers across all sources
- `POST /summarize`: Summarize a single paper
- `POST /related`: Find related papers
- `POST /citations/generate`: Generate citations
- `GET /papers/{id}`: Get paper details
- `POST /papers/compare`: Compare multiple papers

### Example API Usage

```python
import requests

# Search for papers
response = requests.post("http://localhost:8000/search", json={
    "query": "machine learning in healthcare",
    "max_results": 10,
    "sources": ["arxiv", "pubmed"]
})

papers = response.json()["data"]["papers"]
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
isort .
```

### Adding New Data Sources

1. Create a new client in `backend/data_sources/`
2. Implement the standard interface returning `Paper` objects
3. Update the search service to include the new source

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions, please open an issue on the GitHub repository.

## Roadmap

- [ ] Phase 1: Basic ArXiv integration ✅
- [ ] Phase 2: Vector database and semantic search
- [ ] Phase 3: Paper summarization
- [ ] Phase 4: Citation generation
- [ ] Phase 5: PubMed integration
- [ ] Phase 6: Google Scholar integration
- [ ] Phase 7: Advanced features (recommendations, exports)
- [ ] Phase 8: Research timeline visualization
- [ ] Phase 9: Collaborative features
- [ ] Phase 10: Mobile app 