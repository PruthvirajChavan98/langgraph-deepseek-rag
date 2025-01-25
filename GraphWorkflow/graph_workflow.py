from langgraph.graph import StateGraph, START, END
from typing import List, Dict
from utility.answer_grader import grade_answer
from utility.document_grader import grade_document_relevance
from utility.generate import run_rag_chain
from utility.grade_hallucinations import grade_hallucination
from utility.rewrite_questions import rewrite_question
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

class GraphWorkflow:
    """
    Class to encapsulate the graph workflow for document retrieval, generation, and grading.
    """
    def __init__(self, retriever, llm_chat, llm_resoner):
        self.retriever = retriever
        self.llm_chat = llm_chat
        self.llm_resoner = llm_resoner
        self.workflow = self.build_graph_workflow()

    def build_graph_workflow(self):
        """
        Builds the graph workflow.

        Returns:
            StateGraph: The compiled graph workflow.
        """
        workflow = StateGraph(GraphState)

        # Define nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)

        # Define edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        return workflow.compile()

    def retrieve(self, state: GraphState):
        """
        Retrieves relevant documents.

        Args:
            state (GraphState): The current graph state.

        Returns:
            GraphState: Updated state with retrieved documents.
        """
        print("---RETRIEVE---")
        question = state["question"]
        documents = self.retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}

    def grade_documents(self, state: GraphState):
        """
        Grades the relevance of documents to the question.

        Args:
            state (GraphState): The current graph state.

        Returns:
            GraphState: Updated state with filtered relevant documents.
        """
        print("---GRADE DOCUMENTS---")
        question = state["question"]
        documents = state["documents"]
        graded_results = grade_document_relevance(self.llm_chat, self.retriever, question)
        filtered_docs = [doc for doc in documents if doc.page_content in [res["document"] for res in graded_results if res["relevance_score"] == "yes"]]
        return {"documents": filtered_docs, "question": question}

    def generate(self, state: GraphState):
        """
        Generates an answer using retrieved documents.

        Args:
            state (GraphState): The current graph state.

        Returns:
            GraphState: Updated state with generated answer.
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        generation = run_rag_chain(self.llm_resoner, documents, question)

        return {"documents": documents, "question": question, "generation": generation}

    def transform_query(self, state: GraphState):
        """
        Transforms the query to improve relevance.

        Args:
            state (GraphState): The current graph state.

        Returns:
            GraphState: Updated state with transformed question.
        """
        print("---TRANSFORM QUERY---")
        question = state["question"]
        better_question = rewrite_question(self.llm_resoner, question)
        return {"documents": state["documents"], "question": better_question}

    def decide_to_generate(self, state: GraphState):
        """
        Decides whether to generate an answer or transform the query.

        Args:
            state (GraphState): The current graph state.

        Returns:
            str: Decision for next node.
        """
        print("---DECIDE TO GENERATE---")
        if not state["documents"]:
            print("---TRANSFORM QUERY---")
            return "transform_query"
        print("---GENERATE---")
        return "generate"

    def grade_generation_v_documents_and_question(self, state: GraphState):
        """
        Grades the generation against the question and documents.

        Args:
            state (GraphState): The current graph state.

        Returns:
            str: Decision for next node.
        """
        print("---GRADE GENERATION---")
        documents = state["documents"]
        generation = state["generation"]

        hallucination_grade = grade_hallucination(self.llm_chat, documents, generation)
        if hallucination_grade["hallucination_grade"] == "yes":
            answer_grade = grade_answer(self.llm_chat, state["question"], generation)
            if answer_grade["answer_grade"] == "yes":
                return "useful"
            else:
                return "not useful"
        else:
            return "not supported"
        
    async def stream_chunks(self, question: str):
        """
        Streams chunks of the generated answer.

        Args:
            question (str): The question to generate an answer for.

        Yields:
            str: Chunks of the generated answer.
        """
        async for event in self.workflow.astream_events({"question": question}, version="v2"):
            if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node', '') == "generate":
                data = event["data"]
                yield data["chunk"].content
