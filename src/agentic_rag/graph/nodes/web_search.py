from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

from ..state import GraphState


load_dotenv()
web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---- Web Search Node ----")

    question = state["question"]
    documents = state["documents"]

    response = web_search_tool.invoke({"query": question})
    tavily_results = response["results"]

    # Collect the text content of each search result
    contents = []
    for result in tavily_results:
        contents.append(result["content"])

    # Put all the contents together, one per line
    joined_tavily_result = "\n".join(contents)

    web_search_result = Document(
        page_content=joined_tavily_result)

    if documents is not None:
        documents.append(web_search_result)
    else:
        documents = [web_search_result]

if __name__ == "__main__":
    web_search(state=GraphState({"question": "agent memory", "documents": []}))