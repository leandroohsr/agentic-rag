from typing import Any, Dict, List

from agentic_rag.graph.chain.generation import generation_chain
from agentic_rag.graph.chain.retrieval_grader import GradeDocuments, retrieval_grader
from ..state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("----GENERATE----")

    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})

    return {
        "generation": generation,
        "question": question,
        "documents": documents,
    }