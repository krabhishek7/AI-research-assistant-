import gradio as gr
import requests
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import urllib.parse
from datetime import datetime
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
API_BASE_URL = "http://localhost:8000"
SEARCH_SOURCES = ["arxiv", "local", "all"]
CITATION_STYLES = ["APA", "MLA", "Chicago", "BibTeX"]

class APIClient:
    """Client for interacting with the FastAPI backend"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Unable to connect to the API server. Please ensure the backend is running."}
        except requests.exceptions.HTTPError as e:
            return {"success": False, "error": f"HTTP error: {e.response.status_code} - {e.response.text}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def search_papers(self, query: str, source: str = "all", max_results: int = 10) -> Dict[str, Any]:
        """Search for papers"""
        return self._make_request(
            "POST", 
            "/search",
            json={"query": query, "source": source, "max_results": max_results}
        )
    
    def get_paper_by_id(self, paper_id: str) -> Dict[str, Any]:
        """Get paper details by ID"""
        encoded_id = urllib.parse.quote(paper_id, safe='')
        return self._make_request("GET", f"/papers/{encoded_id}")
    
    def get_related_papers(self, paper_id: str, max_results: int = 5) -> Dict[str, Any]:
        """Get related papers"""
        return self._make_request(
            "POST",
            "/related",
            json={"paper_id": paper_id, "max_results": max_results}
        )
    
    def get_suggestions(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Get search suggestions"""
        return self._make_request(
            "POST",
            "/suggestions",
            json={"query": query, "max_results": max_results}
        )
    
    def process_paper(self, paper_id: str) -> Dict[str, Any]:
        """Process paper for full text extraction"""
        return self._make_request(
            "POST",
            "/process",
            json={"paper_id": paper_id}
        )
    
    def compare_papers(self, paper_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple papers"""
        return self._make_request(
            "POST",
            "/compare",
            json={"paper_ids": paper_ids}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return self._make_request("GET", "/stats")
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        return self._make_request("GET", "/health")
    
    def summarize_paper(self, paper_id: str = None, text: str = None, method: str = "abstractive", max_length: int = None) -> Dict[str, Any]:
        """Summarize a paper or text"""
        data = {
            "method": method
        }
        if paper_id:
            data["paper_id"] = paper_id
        if text:
            data["text"] = text
        if max_length:
            data["max_length"] = max_length
        
        return self._make_request("POST", "/summarize", json=data)
    
    def summarize_abstract(self, abstract: str, max_length: int = 100) -> Dict[str, Any]:
        """Summarize an abstract"""
        return self._make_request("POST", "/summarize/abstract", json={"abstract": abstract, "max_length": max_length})
    
    def extract_key_findings(self, paper_id: str = None, text: str = None, max_findings: int = 5) -> Dict[str, Any]:
        """Extract key findings from a paper"""
        data = {
            "max_findings": max_findings
        }
        if paper_id:
            data["paper_id"] = paper_id
        if text:
            data["text"] = text
        
        return self._make_request("POST", "/extract/key-findings", json=data)
    
    def get_summarizer_info(self) -> Dict[str, Any]:
        """Get summarizer information"""
        return self._make_request("GET", "/summarizer/info")

# Initialize API client
api_client = APIClient()

def format_paper_results(papers: List[Dict[str, Any]]) -> str:
    """Format paper results for display"""
    if not papers:
        return "No papers found."
    
    formatted_results = []
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'Unknown Title')
        authors = ', '.join(paper.get('authors', [])) if paper.get('authors') else 'Unknown Authors'
        abstract = paper.get('abstract', 'No abstract available')
        published = paper.get('published', 'Unknown Date')
        source = paper.get('source', 'Unknown Source')
        score = paper.get('score', 0)
        
        # Truncate abstract if too long
        if len(abstract) > 300:
            abstract = abstract[:300] + "..."
        
        formatted_results.append(f"""
**{i}. {title}**
- **Authors:** {authors}
- **Published:** {published}
- **Source:** {source}
- **Relevance Score:** {score:.3f}
- **Abstract:** {abstract}
- **ID:** {paper.get('id', 'N/A')}
---
""")
    
    return "\n".join(formatted_results)

def search_papers_interface(query: str, source: str, max_results: int):
    """Interface for searching papers"""
    if not query.strip():
        return "Please enter a search query.", ""
    
    result = api_client.search_papers(query, source, max_results)
    
    if "error" in result:
        return f"Error: {result['error']}", ""
    
    papers = result.get("papers", [])
    stats = result.get("stats", {})
    
    # Format results
    formatted_results = format_paper_results(papers)
    
    # Create stats summary
    stats_summary = f"""
**Search Statistics:**
- Total papers found: {stats.get('total_papers', 0)}
- Search time: {stats.get('search_time', 0):.2f} seconds
- Source: {stats.get('source', 'Unknown')}
"""
    
    return formatted_results, stats_summary

def get_paper_details(paper_id: str):
    """Get detailed information about a specific paper"""
    if not paper_id.strip():
        return "Please enter a paper ID."
    
    result = api_client.get_paper_by_id(paper_id)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    paper = result.get("paper", {})
    
    # Format paper details
    title = paper.get('title', 'Unknown Title')
    authors = ', '.join(paper.get('authors', [])) if paper.get('authors') else 'Unknown Authors'
    abstract = paper.get('abstract', 'No abstract available')
    published = paper.get('published', 'Unknown Date')
    source = paper.get('source', 'Unknown Source')
    url = paper.get('url', 'No URL available')
    
    return f"""
**{title}**

**Authors:** {authors}

**Published:** {published}

**Source:** {source}

**URL:** {url}

**Abstract:**
{abstract}

**ID:** {paper.get('id', 'N/A')}
"""

def find_related_papers(paper_id: str, max_results: int):
    """Find papers related to a given paper"""
    if not paper_id.strip():
        return "Please enter a paper ID."
    
    result = api_client.get_related_papers(paper_id, max_results)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    papers = result.get("related_papers", [])
    
    if not papers:
        return "No related papers found."
    
    return format_paper_results(papers)

def get_search_suggestions(query: str, max_results: int):
    """Get search suggestions"""
    if not query.strip():
        return "Please enter a search query."
    
    result = api_client.get_suggestions(query, max_results)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    suggestions = result.get("suggestions", [])
    
    if not suggestions:
        return "No suggestions found."
    
    formatted_suggestions = []
    for i, suggestion in enumerate(suggestions, 1):
        formatted_suggestions.append(f"{i}. {suggestion}")
    
    return "\n".join(formatted_suggestions)

def compare_papers_interface(paper_ids_text: str):
    """Compare multiple papers"""
    if not paper_ids_text.strip():
        return "Please enter paper IDs separated by commas."
    
    # Parse paper IDs
    paper_ids = [pid.strip() for pid in paper_ids_text.split(',') if pid.strip()]
    
    if len(paper_ids) < 2:
        return "Please enter at least 2 paper IDs for comparison."
    
    result = api_client.compare_papers(paper_ids)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    comparison = result.get("comparison", {})
    
    # Format comparison results
    common_keywords = comparison.get("common_keywords", [])
    summary = comparison.get("summary", "No summary available")
    
    formatted_comparison = f"""
**Paper Comparison Results**

**Common Keywords:** {', '.join(common_keywords) if common_keywords else 'None found'}

**Summary:**
{summary}

**Papers Compared:** {len(paper_ids)}
"""
    
    return formatted_comparison

def process_paper_interface(paper_id: str):
    """Process a paper for full text extraction"""
    if not paper_id.strip():
        return "Please enter a paper ID."
    
    result = api_client.process_paper(paper_id)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    return f"Paper processed successfully! Status: {result.get('status', 'Unknown')}"

def generate_citation(paper_id: str, style: str):
    """Generate citation for a paper"""
    if not paper_id.strip():
        return "Please enter a paper ID."
    
    # Get paper details first
    paper_result = api_client.get_paper_by_id(paper_id)
    
    if "error" in paper_result:
        return f"Error: {paper_result['error']}"
    
    paper = paper_result.get("paper", {})
    
    # Generate citation based on style
    title = paper.get('title', 'Unknown Title')
    authors = paper.get('authors', ['Unknown Author'])
    published = paper.get('published', 'Unknown Date')
    url = paper.get('url', 'No URL')
    
    # Format authors
    if len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]} & {authors[1]}"
    else:
        author_str = f"{authors[0]} et al."
    
    # Extract year from published date
    try:
        year = datetime.strptime(published, "%Y-%m-%d").year
    except:
        year = "Unknown"
    
    # Generate citation based on style
    if style == "APA":
        citation = f"{author_str} ({year}). {title}. Retrieved from {url}"
    elif style == "MLA":
        citation = f"{author_str}. \"{title}.\" {year}. Web. {datetime.now().strftime('%d %b %Y')}."
    elif style == "Chicago":
        citation = f"{author_str}. \"{title}.\" Accessed {datetime.now().strftime('%B %d, %Y')}. {url}."
    elif style == "BibTeX":
        citation = f"""@article{{{paper.get('id', 'unknown')},
    title = {{{title}}},
    author = {{{'; '.join(authors)}}},
    year = {{{year}}},
    url = {{{url}}}
}}"""
    else:
        citation = f"Unknown citation style: {style}"
    
    return citation

def summarize_paper_interface(paper_id: str, method: str, max_length: int):
    """Interface for summarizing papers"""
    if not paper_id.strip():
        return "Please enter a paper ID."
    
    result = api_client.summarize_paper(paper_id=paper_id, method=method, max_length=max_length)
    
    if not result.get("success", False):
        return f"Error: {result.get('error', 'Unknown error')}"
    
    data = result.get("data", {})
    
    return f"""
**Paper Summarization Results**

**Summary:**
{data.get('summary', 'No summary available')}

**Method:** {data.get('method', 'Unknown')}
**Original length:** {data.get('original_length', 0)} words
**Summary length:** {data.get('summary_length', 0)} words
**Compression ratio:** {data.get('compression_ratio', 0):.2%}
**Confidence score:** {data.get('confidence_score', 0):.2%}

**Key Points:**
{chr(10).join(f"• {point}" for point in data.get('key_points', []))}

**Model:** {data.get('summarizer_model', 'Unknown')}
"""

def summarize_text_interface(text: str, method: str, max_length: int):
    """Interface for summarizing text"""
    if not text.strip():
        return "Please enter text to summarize."
    
    result = api_client.summarize_paper(text=text, method=method, max_length=max_length)
    
    if not result.get("success", False):
        return f"Error: {result.get('error', 'Unknown error')}"
    
    data = result.get("data", {})
    
    return f"""
**Text Summarization Results**

**Summary:**
{data.get('summary', 'No summary available')}

**Method:** {data.get('method', 'Unknown')}
**Original length:** {data.get('original_length', 0)} words
**Summary length:** {data.get('summary_length', 0)} words
**Compression ratio:** {data.get('compression_ratio', 0):.2%}
**Confidence score:** {data.get('confidence_score', 0):.2%}

**Key Points:**
{chr(10).join(f"• {point}" for point in data.get('key_points', []))}
"""

def extract_key_findings_interface(paper_id: str, max_findings: int):
    """Interface for extracting key findings"""
    if not paper_id.strip():
        return "Please enter a paper ID."
    
    result = api_client.extract_key_findings(paper_id=paper_id, max_findings=max_findings)
    
    if not result.get("success", False):
        return f"Error: {result.get('error', 'Unknown error')}"
    
    data = result.get("data", {})
    key_findings = data.get("key_findings", [])
    
    if not key_findings:
        return "No key findings found."
    
    formatted_findings = []
    for i, finding in enumerate(key_findings, 1):
        formatted_findings.append(f"{i}. {finding}")
    
    return f"""
**Key Findings**

{chr(10).join(formatted_findings)}

**Total findings:** {data.get('total_findings', 0)}
**Text length:** {data.get('text_length', 0)} words
"""

def get_system_stats():
    """Get system statistics"""
    result = api_client.get_stats()
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    stats = result.get("stats", {})
    
    # Get summarizer info
    summarizer_result = api_client.get_summarizer_info()
    summarizer_info = summarizer_result.get("data", {}) if "data" in summarizer_result else {}
    
    return f"""
**System Statistics**

**Database Stats:**
- Total papers: {stats.get('total_papers', 0)}
- ArXiv papers: {stats.get('arxiv_papers', 0)}
- Local papers: {stats.get('local_papers', 0)}

**Configuration:**
- Embeddings model: {stats.get('embeddings_model', 'Unknown')}
- Database status: {stats.get('database_status', 'Unknown')}

**Summarizer Info:**
- Model: {summarizer_info.get('model_name', 'Unknown')}
- Device: {summarizer_info.get('device', 'Unknown')}
- Max length: {summarizer_info.get('max_length', 'Unknown')}
- Model loaded: {summarizer_info.get('model_loaded', False)}

**API Health:** {api_client.health_check().get('success', False)}
"""

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(title="Academic Research Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔬 Academic Research Assistant")
        gr.Markdown("A comprehensive tool for searching, analyzing, and managing academic papers")
        
        with gr.Tabs():
            # Search Papers Tab
            with gr.TabItem("🔍 Search Papers"):
                gr.Markdown("### Search for academic papers across multiple sources")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        search_query = gr.Textbox(
                            label="Search Query",
                            placeholder="Enter your search query (e.g., 'machine learning', 'quantum computing')",
                            lines=2
                        )
                        search_source = gr.Dropdown(
                            choices=SEARCH_SOURCES,
                            label="Source",
                            value="all"
                        )
                        search_max_results = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            label="Max Results",
                            value=10
                        )
                        search_btn = gr.Button("Search Papers", variant="primary")
                    
                    with gr.Column(scale=1):
                        suggestions_btn = gr.Button("Get Suggestions")
                        suggestions_output = gr.Textbox(
                            label="Search Suggestions",
                            lines=5,
                            interactive=False
                        )
                
                search_results = gr.Textbox(
                    label="Search Results",
                    lines=20,
                    interactive=False
                )
                search_stats = gr.Textbox(
                    label="Search Statistics",
                    lines=3,
                    interactive=False
                )
                
                search_btn.click(
                    fn=search_papers_interface,
                    inputs=[search_query, search_source, search_max_results],
                    outputs=[search_results, search_stats]
                )
                
                suggestions_btn.click(
                    fn=get_search_suggestions,
                    inputs=[search_query, gr.Number(value=5, visible=False)],
                    outputs=[suggestions_output]
                )
            
            # Paper Details Tab
            with gr.TabItem("📄 Paper Details"):
                gr.Markdown("### Get detailed information about a specific paper")
                
                paper_id_input = gr.Textbox(
                    label="Paper ID",
                    placeholder="Enter paper ID (e.g., from search results)",
                    lines=1
                )
                
                with gr.Row():
                    details_btn = gr.Button("Get Details", variant="primary")
                    process_btn = gr.Button("Process Paper", variant="secondary")
                
                paper_details = gr.Textbox(
                    label="Paper Details",
                    lines=15,
                    interactive=False
                )
                
                process_status = gr.Textbox(
                    label="Processing Status",
                    lines=2,
                    interactive=False
                )
                
                details_btn.click(
                    fn=get_paper_details,
                    inputs=[paper_id_input],
                    outputs=[paper_details]
                )
                
                process_btn.click(
                    fn=process_paper_interface,
                    inputs=[paper_id_input],
                    outputs=[process_status]
                )
            
            # Related Papers Tab
            with gr.TabItem("🔗 Related Papers"):
                gr.Markdown("### Find papers related to a specific paper")
                
                related_paper_id = gr.Textbox(
                    label="Paper ID",
                    placeholder="Enter paper ID to find related papers",
                    lines=1
                )
                related_max_results = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    label="Max Results",
                    value=5
                )
                related_btn = gr.Button("Find Related Papers", variant="primary")
                
                related_results = gr.Textbox(
                    label="Related Papers",
                    lines=15,
                    interactive=False
                )
                
                related_btn.click(
                    fn=find_related_papers,
                    inputs=[related_paper_id, related_max_results],
                    outputs=[related_results]
                )
            
            # Compare Papers Tab
            with gr.TabItem("⚖️ Compare Papers"):
                gr.Markdown("### Compare multiple papers")
                
                compare_paper_ids = gr.Textbox(
                    label="Paper IDs",
                    placeholder="Enter paper IDs separated by commas (e.g., id1, id2, id3)",
                    lines=3
                )
                compare_btn = gr.Button("Compare Papers", variant="primary")
                
                comparison_results = gr.Textbox(
                    label="Comparison Results",
                    lines=15,
                    interactive=False
                )
                
                compare_btn.click(
                    fn=compare_papers_interface,
                    inputs=[compare_paper_ids],
                    outputs=[comparison_results]
                )
            
            # Citation Generator Tab
            with gr.TabItem("📖 Citation Generator"):
                gr.Markdown("### Generate citations for papers")
                
                citation_paper_id = gr.Textbox(
                    label="Paper ID",
                    placeholder="Enter paper ID to generate citation",
                    lines=1
                )
                citation_style = gr.Dropdown(
                    choices=CITATION_STYLES,
                    label="Citation Style",
                    value="APA"
                )
                citation_btn = gr.Button("Generate Citation", variant="primary")
                
                citation_output = gr.Textbox(
                    label="Generated Citation",
                    lines=8,
                    interactive=False
                )
                
                citation_btn.click(
                    fn=generate_citation,
                    inputs=[citation_paper_id, citation_style],
                    outputs=[citation_output]
                )
            
            # Summarization Tab
            with gr.TabItem("📝 Summarization"):
                gr.Markdown("### AI-powered paper summarization and key findings extraction")
                
                with gr.Tabs():
                    # Paper Summarization
                    with gr.TabItem("📄 Paper Summary"):
                        gr.Markdown("#### Summarize a paper by ID")
                        
                        with gr.Row():
                            with gr.Column():
                                paper_sum_id = gr.Textbox(
                                    label="Paper ID",
                                    placeholder="Enter paper ID from search results",
                                    lines=1
                                )
                                paper_sum_method = gr.Dropdown(
                                    choices=["abstractive", "extractive"],
                                    label="Summarization Method",
                                    value="abstractive"
                                )
                                paper_sum_length = gr.Slider(
                                    minimum=50,
                                    maximum=500,
                                    step=25,
                                    label="Max Summary Length",
                                    value=200
                                )
                                paper_sum_btn = gr.Button("Summarize Paper", variant="primary")
                        
                        paper_sum_output = gr.Textbox(
                            label="Paper Summary",
                            lines=15,
                            interactive=False
                        )
                        
                        paper_sum_btn.click(
                            fn=summarize_paper_interface,
                            inputs=[paper_sum_id, paper_sum_method, paper_sum_length],
                            outputs=[paper_sum_output]
                        )
                    
                    # Text Summarization
                    with gr.TabItem("📝 Text Summary"):
                        gr.Markdown("#### Summarize any text directly")
                        
                        with gr.Row():
                            with gr.Column():
                                text_sum_input = gr.Textbox(
                                    label="Text to Summarize",
                                    placeholder="Paste your text here...",
                                    lines=8
                                )
                                text_sum_method = gr.Dropdown(
                                    choices=["abstractive", "extractive"],
                                    label="Summarization Method",
                                    value="abstractive"
                                )
                                text_sum_length = gr.Slider(
                                    minimum=50,
                                    maximum=500,
                                    step=25,
                                    label="Max Summary Length",
                                    value=200
                                )
                                text_sum_btn = gr.Button("Summarize Text", variant="primary")
                        
                        text_sum_output = gr.Textbox(
                            label="Text Summary",
                            lines=15,
                            interactive=False
                        )
                        
                        text_sum_btn.click(
                            fn=summarize_text_interface,
                            inputs=[text_sum_input, text_sum_method, text_sum_length],
                            outputs=[text_sum_output]
                        )
                    
                    # Key Findings
                    with gr.TabItem("🔍 Key Findings"):
                        gr.Markdown("#### Extract key findings from a paper")
                        
                        with gr.Row():
                            with gr.Column():
                                findings_paper_id = gr.Textbox(
                                    label="Paper ID",
                                    placeholder="Enter paper ID from search results",
                                    lines=1
                                )
                                findings_count = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    step=1,
                                    label="Number of Key Findings",
                                    value=5
                                )
                                findings_btn = gr.Button("Extract Key Findings", variant="primary")
                        
                        findings_output = gr.Textbox(
                            label="Key Findings",
                            lines=12,
                            interactive=False
                        )
                        
                        findings_btn.click(
                            fn=extract_key_findings_interface,
                            inputs=[findings_paper_id, findings_count],
                            outputs=[findings_output]
                        )
            
            # System Stats Tab
            with gr.TabItem("📊 System Stats"):
                gr.Markdown("### System statistics and health")
                
                stats_btn = gr.Button("Get System Stats", variant="primary")
                
                stats_output = gr.Textbox(
                    label="System Statistics",
                    lines=15,
                    interactive=False
                )
                
                stats_btn.click(
                    fn=get_system_stats,
                    inputs=[],
                    outputs=[stats_output]
                )
                
                # Auto-load stats on page load
                demo.load(
                    fn=get_system_stats,
                    inputs=[],
                    outputs=[stats_output]
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("**Note:** Make sure the FastAPI backend is running on `http://localhost:8000` for full functionality.")
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    ) 