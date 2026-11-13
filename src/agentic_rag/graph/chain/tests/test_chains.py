from dotenv import load_dotenv


load_dotenv()


from agentic_rag.graph.chain import retrieval_grader, GradeDocuments

from agentic_rag.ingestion import retriever
from agentic_rag.graph.chain import generation_chain


def test_retrieval_grader_answer_yes():
    question = "agent memory"

    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": question}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": "how to make pizaa"}
    )

    assert res.binary_score == "no"


def test_generation_chain():
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    print(f"Generation chain output: {generation}")