from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def rewrite_question(llm_resoner, question):
    """
    Rewrites a given question to optimize it for vectorstore retrieval.

    Args:
        llm_resoner: The LLM reasoning component.
        question (str): The initial user question.

    Returns:
        str: The improved version of the question.
    """
    # Define the system prompt for question re-writing
    system = """You are a question re-writer that converts an input question to a better version that is optimized 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    
    # Define the re-write prompt
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question."
                "Make sure to give the improved question ONLY in your output.",
            ),
        ]
    )

    # Combine the re-write prompt with the LLM reasoner and output parser
    question_rewriter = re_write_prompt | llm_resoner | StrOutputParser()

    # Invoke the re-writer with the input question
    improved_question = question_rewriter.invoke({"question": question})

    return improved_question


# Usage Example:
# improved = rewrite_question(llm_resoner, "What is the best way to learn machine learning?")
# print(improved)
