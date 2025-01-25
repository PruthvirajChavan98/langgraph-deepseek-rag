import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from GraphWorkflow.graph_workflow import GraphWorkflow

class DocumentProcessingPipeline:
    def __init__(self, pdf_path, embedding_model, chat_model, reasoner_model, vectorstore_base_path="./vectorstores"):
        """
        Initializes the DocumentProcessingWorkflow with either pre-initialized or new components.

        Args:
            pdf_path (str): The path to the PDF file.
            embedding_model (str): The model name for embeddings.
            chat_model (str): The model name for chat.
            reasoner_model (str): The model name for reasoner.
            vectorstore_base_path (str): The base directory to store the vectorstore for each PDF.
        """
        self.pdf_path = pdf_path
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.reasoner_model = reasoner_model

        self.loader = PDFPlumberLoader(self.pdf_path)

        # Generate a unique vectorstore path based on the PDF name
        pdf_name_without_extension = os.path.basename(pdf_path).replace(".pdf", "")
        self.vectorstore_path = os.path.join(vectorstore_base_path, pdf_name_without_extension)

        # If vectorstore and retriever are provided, use them; otherwise, initialize them
        self.vectorstore = self.create_or_load_vectorstore()
        self.retriever = self.vectorstore.as_retriever()
        self.workflow = self.create_workflow(self.retriever)

    def load_and_split_documents(self, chunk_size=100, chunk_overlap=50):
        """
        Loads and splits the documents from the PDF.

        Args:
            chunk_size (int): The size of the chunks for splitting the documents.
            chunk_overlap (int): The overlap between chunks.

        Returns:
            list: A list of split documents.
        """
        docs = self.loader.load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(docs)

    def create_or_load_vectorstore(self, documents=None):
        """
        If the vectorstore directory exists and is non-empty, 
        load the existing store. Otherwise, read PDF, chunk it, 
        and create a new one.

        Args:
            documents (optional): If provided, it is used to create a new vectorstore.

        Returns:
            Chroma: The initialized or loaded vectorstore.
        """
        # Check if vectorstore exists and is non-empty
        if (
            os.path.exists(self.vectorstore_path)
            and os.path.isdir(self.vectorstore_path)
            and os.listdir(self.vectorstore_path)
        ):
            print(f"[DocumentProcessingWorkflow] Loading existing vectorstore from: {self.vectorstore_path}")
            return Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=self.embedding_model
            )
        
        # If no existing index, read PDF and create/persist a new index
        print(f"[DocumentProcessingWorkflow] No existing vectorstore found; creating a new one at: {self.vectorstore_path}")
        
        if not documents:
            # If documents aren't provided, load them from the PDF
            docs = self.loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            documents = splitter.split_documents(docs)
        
        # Create and persist the new vectorstore
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.vectorstore_path
        )
        
        return vectorstore

    def create_workflow(self, retriever):
        """
        Creates the workflow for the entire process.

        Args:
            retriever (ChromaRetriever): The retriever for document retrieval.

        Returns:
            GraphWorkflow: The initialized workflow.
        """
        return GraphWorkflow(
            llm_chat=self.chat_model, 
            llm_resoner=self.reasoner_model, 
            retriever=retriever
        )

    async def run_workflow(self, query):
        """
        Runs the workflow asynchronously.

        Args:
            query (str): The user's question.

        Yields:
            str: Chunks of the generated answer from the workflow.
        """
        async for i in self.workflow.stream_chunks(query):
            yield i