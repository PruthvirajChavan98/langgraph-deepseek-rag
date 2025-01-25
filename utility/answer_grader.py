from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class GradeAnswer(BaseModel):
    """Binary score to assess whether the answer addresses the question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


def grade_answer(llm_chat, question, generation):
    """
    Grades whether an LLM-generated answer addresses the user's question.

    Args:
        llm_chat: The LLM instance with structured output capabilities.
        question (str): The user's question.
        generation (str): The LLM's generated answer.

    Returns:
        dict: A dictionary containing the question, generation, and its grade ('yes' or 'no').
    """
    # Define the structured LLM grader
    structured_llm_grader = llm_chat.with_structured_output(GradeAnswer)

    # Define the system prompt for answer grading
    system = """You are a grader assessing whether an answer addresses / resolves a question. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
    
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    # Combine the prompt and the structured LLM grader
    answer_grader = answer_prompt | structured_llm_grader

    # Invoke the answer grader
    grading_result = answer_grader.invoke({"question": question, "generation": generation})

    return {
        "question": question,
        "generation": generation,
        "answer_grade": grading_result.binary_score
    }


# Usage Example:
# graded_result = grade_answer(llm_chat, "What is AI?", "AI is artificial intelligence.")
# print(graded_result)
