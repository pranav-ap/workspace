import os
from typing import Dict, List

from fastmcp import FastMCP
from tavily import TavilyClient

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("Please set the TAVILY_API_KEY environment variable.")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

mcp = FastMCP(name="ArxivExplorer")
print("ArxivExplorer server initialized.")

# To inspect this server, use the command:
# npx @modelcontextprotocol/inspector


@mcp.resource("resource://ai/arxiv_topics")
def arxiv_topics() -> List[str]:
    return [
        "Transformer interpretability",
        "Efficient large-scale model training",
        "Federated learning privacy",
        "Neural network pruning",
    ]


@mcp.tool(annotations={"title": "Search Arxiv"})
def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    Queries ArXiv via Tavily, returning title + link for each paper,
    and *only* ArXiv results.
    """
    resp = tavily.search(
        query=f"site:arxiv.org {query}",
        max_results=max_results
    )

    return [
        {
            "title": r["title"].strip(),
            "url": r["url"],
        } for r in resp.get("results", [])
    ]


@mcp.tool(annotations={"title": "Summarize Paper"})
def summarize_paper(paper_url: str) -> str:
    """
    Returns a one-paragraph summary of the paper at the given URL.
    """
    prompt = f"Summarize the key contributions of this ArXiv paper: {paper_url}"
    return tavily.qna_search(query=prompt)


@mcp.prompt
def explore_topic_prompt(topic: str) -> str:
    return (
        f"I want to explore recent work on '{topic}'.\n"
        f"1. Call the 'Search Arxiv' tool to find the 5 most recent papers.\n"
        f"2. For each paper URL, call 'Summarize Paper' to extract its key contributions.\n"
        f"3. Combine all summaries into an overview report."
    )


if __name__ == "__main__":
    print("\nStarting ArxivExplorer Server...")
    mcp.run(transport='http')
