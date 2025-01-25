from langchain import hub
from langchain_core.output_parsers import StrOutputParser


def run_rag_chain(llm_resoner, docs, question):
    """
    Executes a Retrieval-Augmented Generation (RAG) chain.

    Args:
        llm_resoner: The LLM reasoning component.
        docs (list): A list of documents retrieved as context.
        question (str): The user question to generate a response for.

    Returns:
        str: The generated response from the RAG chain.
    """
    # Pull the prompt from LangChain hub
    prompt = hub.pull("rlm/rag-prompt")

    # Post-processing: Format the documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Format documents for the chain input
    formatted_context = format_docs(docs)

    # Chain: Prompt → LLM Resoner → Output Parser
    rag_chain = prompt | llm_resoner | StrOutputParser()

    # Run the chain with the context and question
    generation = rag_chain.invoke({"context": formatted_context, "question": question})

    return generation