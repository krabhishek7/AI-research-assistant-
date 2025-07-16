# Academic Research Assistant 🤖

An intelligent chatbot designed to streamline academic research by leveraging Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). This tool helps researchers discover, summarize, compare, and cite academic papers from major scientific databases.

The system integrates with ArXiv, PubMed, and Google Scholar to fetch papers, processes them into a searchable vector database using ChromaDB, and provides an interactive Gradio interface for a seamless user experience.

-----

## ✨ Features

  * **📚 Multi-Source Paper Search**: Simultaneously search for papers across ArXiv, PubMed, and Google Scholar by keyword, author, or topic.
  * **🔍 Advanced Semantic Search**: Go beyond keywords. Find conceptually similar papers using sentence-transformer embeddings (`allenai/specter`). Includes re-ranking with cross-encoders for higher accuracy.
  * **📝 AI-Powered Summarization**: Generate concise abstractive and extractive summaries of papers using Hugging Face models (BART, T5). Extract key points like methodology, results, and limitations.
  * **🧠 Intelligent Q\&A with LlamaIndex**: Ask complex questions across a collection of papers. The system uses LlamaIndex query engines to synthesize answers from multiple sources.
  * **✍️ Automatic Citation Generation**: Instantly generate citations for papers in various formats, including APA, MLA, Chicago, and BibTeX.
  * **📊 Comparative Analysis**: Select multiple papers and generate a side-by-side comparison of their key findings, methodologies, and conclusions.
  * **🌐 Knowledge Graph Visualization**: Understand the relationships between papers, authors, and concepts through a visualized knowledge graph.
  * **🖥️ Interactive Gradio UI**: A user-friendly interface with dedicated tabs for searching, summarizing, managing a personal library, and generating citations.

-----

## 🛠️ Technology Stack

  * **Backend**: FastAPI
  * **Frontend**: Gradio
  * **Vector Database**: ChromaDB
  * **AI/NLP Frameworks**: LlamaIndex, Hugging Face (Transformers, Sentence-Transformers)
  * **Data Sources**: ArXiv API, PubMed API, Google Scholar Scraper
  * **PDF Processing**: `pdfplumber`
  * **Language**: Python 3.9+

-----

## 📂 Project Structure

```
academic-research-assistant/
├── backend/
│   ├── api/
│   │   └── main.py         # FastAPI endpoints
│   ├── data_sources/
│   │   ├── arxiv_client.py   # ArXiv API integration
│   │   ├── pubmed_client.py  # PubMed API integration
│   │   └── scholar_client.py # Google Scholar scraper
│   ├── database/
│   │   ├── chromadb_client.py # ChromaDB vector store operations
│   │   └── models.py        # Data models for papers
│   ├── processing/
│   │   ├── paper_processor.py # Extract and clean paper content
│   │   ├── embeddings.py      # Generate embeddings using Hugging Face
│   │   └── summarizer.py      # Paper summarization logic
│   └── services/
│       ├── search_service.py    # Semantic search implementation
│       ├── citation_service.py  # Citation formatting
│       └── recommendation.py  # Related paper suggestions
├── frontend/
│   └── app.py              # Gradio interface
├── config/
│   └── settings.py         # Configuration management
├── .env.example            # Environment variable template
└── requirements.txt        # Python dependencies
```

-----

## 🚀 Getting Started

### 1\. Prerequisites

  * Python 3.9+
  * Git

### 2\. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/academic-research-assistant.git
    cd academic-research-assistant
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory by copying the example file:

    ```bash
    cp .env.example .env
    ```

    Now, edit the `.env` file and add your API keys and configuration settings.

### 3\. Running the Application

1.  **Start the Backend API (FastAPI):**
    Open a terminal and run:

    ```bash
    uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
    ```

    The API will be available at `http://localhost:8000/docs`.

2.  **Launch the Frontend UI (Gradio):**
    Open a second terminal and run:

    ```bash
    python frontend/app.py
    ```

    The user interface will be accessible at `http://127.0.0.1:7860`.

-----

## ⚙️ Configuration

The following environment variables must be set in your `.env` file:

| Variable                  | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| `OPENAI_API_KEY`          | Optional. Your API key for OpenAI models.              |
| `HUGGINGFACE_TOKEN`       | Your Hugging Face Hub token for accessing private models. |
| `CHROMA_PERSIST_DIRECTORY`| Path to the directory for persisting the ChromaDB database. Default: `./chroma_db` |
| `ARXIV_MAX_RESULTS`       | Maximum number of results to fetch from ArXiv per query. Default: `100` |
| `PUBMED_API_KEY`          | Optional. Your NCBI API key for PubMed access.         |

-----

## Endpoints API

The backend provides the following RESTful API endpoints:

| Method | Endpoint              | Description                                        |
| :----- | :-------------------- | :------------------------------------------------- |
| `POST` | `/search`             | Searches for papers across all integrated sources. |
| `POST` | `/summarize`          | Generates a summary for a single paper via URL/ID. |
| `POST` | `/related`            | Finds and returns papers related to a given paper. |
| `POST` | `/citations/generate` | Generates a citation for a paper in a specific style. |
| `GET`  | `/papers/{id}`        | Retrieves detailed information for a specific paper. |
| `POST` | `/papers/compare`     | Compares two or more papers side-by-side.        |

-----

## 🤝 Contributing

Contributions are welcome\! If you'd like to contribute, please fork the repository and create a pull request. You can also open an issue with the "enhancement" tag to suggest new features.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

-----

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
