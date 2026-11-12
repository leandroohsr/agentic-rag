from typing import Any, Dict, List, Optional, Tuple


from ..state import GraphState
from ...ingestion import retriever


def retrive(state: GraphState) -> Dict[str, Any]:
    print("----RETRIEVE----")

    question = state["question"]

    documents = retriever.invoke(question)

    return {
        "documents": documents,
        "question": question,
    }