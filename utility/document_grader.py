from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def grade_document_relevance(llm_chat, retriever, question):
    """
    Grades the relevance of a retrieved document to a user question using a structured LLM.

    Args:
        llm_chat: The LLM instance with structured output capabilities.
        retriever: The document retriever.
        question (str): The user's query.

    Returns:
        dict: A dictionary with the document content and its binary relevance score.
    """
    # Define the structured LLM grader
    structured_llm_grader = llm_chat.with_structured_output(GradeDocuments)

    # Define the system prompt for grading
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    # Combine the prompt and the structured LLM grader
    retrieval_grader = grade_prompt | structured_llm_grader

    # Retrieve documents
    docs = retriever.get_relevant_documents(question)

    # Grade each document
    graded_results = []
    for doc in docs:
        document_text = doc.page_content
        grading_result = retrieval_grader.invoke({"question": question, "document": document_text})
        graded_results.append({
            "document": document_text,
            "relevance_score": grading_result.binary_score
        })

    return graded_results