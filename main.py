from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import os
import sqlite3
from DocumentProcessingPipeline.document_processing_pipeline import DocumentProcessingPipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI()

# Add CORS middleware to your FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize database
DB_PATH = "pipelines.db"
UPLOAD_FOLDER = "./uploads"
VECTORSTORE_BASE_PATH = "./vectorstores"  # Base folder for all vectorstore data
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_BASE_PATH, exist_ok=True)  # Ensure the vectorstore directory exists

# Pre-initialize components (embeddings, vectorstore, retriever)
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/modernbert-embed-base", cache_folder="./saved_model")
llm_chat = ChatOpenAI(model="deepseek-chat", temperature=0, openai_api_key="YOUR_DEEPSEEK_API_KEY", openai_api_base='https://api.deepseek.com')
llm_resoner = ChatOpenAI(model="deepseek-reasoner", temperature=0, openai_api_key="YOUR_DEEPSEEK_API_KEY", openai_api_base='https://api.deepseek.com')

# Pydantic model
class AskQuestionRequest(BaseModel):
    question: str = Field(..., description="Question about the PDF.")
    pdf_name: str = Field(..., description="Name of the uploaded PDF.")

# Database helper functions
def init_db():
    """Initialize the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipelines (
                pdf_name TEXT PRIMARY KEY,
                vectorstore_path TEXT,
                pdf_path TEXT
            )
        """)
        conn.commit()

def store_pipeline_metadata(pdf_name: str, pdf_path: str, vectorstore_path: str):
    """Store pipeline metadata in SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO pipelines (pdf_name, pdf_path, vectorstore_path)
            VALUES (?, ?, ?)
        """, (pdf_name, pdf_path, vectorstore_path))
        conn.commit()

def get_pipeline_metadata(pdf_name: str):
    """Retrieve pipeline metadata from SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pdf_path, vectorstore_path FROM pipelines WHERE pdf_name = ?
        """, (pdf_name,))
        return cursor.fetchone()

# Initialize the DB
init_db()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            return {"error": "Only PDF files are supported."}

        # Save the file in the uploads folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Generate a unique vectorstore path outside the uploads folder, in the vectorstore base path
        vectorstore_path = os.path.join(VECTORSTORE_BASE_PATH, file.filename.replace(".pdf", ""))

        # Store metadata in SQLite
        store_pipeline_metadata(file.filename, file_path, vectorstore_path)

        return {"message": "File uploaded successfully.", "pdf_path": file_path, "vectorstore_path": vectorstore_path}
    except Exception as e:
        return {"error": f"Error uploading file: {str(e)}"}


@app.post("/ask")
async def ask_question(request: Request, params: AskQuestionRequest):
    try:
        
        # Rest of the logic
        metadata = get_pipeline_metadata(params.pdf_name)
        if not metadata:
            return {"error": "No pipeline found for the provided PDF name."}

        pdf_path, vectorstore_path = metadata

        doc_rag = DocumentProcessingPipeline(
            pdf_path=pdf_path,
            embedding_model=embeddings,
            chat_model=llm_chat,
            reasoner_model=llm_resoner,
            vectorstore_base_path=vectorstore_path
        )

        return StreamingResponse(doc_rag.run_workflow(params.question), media_type="text/event-stream")
    except Exception as e:
        raise e
