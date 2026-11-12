from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: bool
    documents: List[str]