from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from DocumentProcessingPipeline.document_processing_pipeline import DocumentProcessingPipeline
from utility.db_utility import init_db, get_pipeline_metadata, store_pipeline_metadata

load_dotenv()

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

UPLOAD_FOLDER = "./uploads"
VECTORSTORE_BASE_PATH = "./vectorstores"  # Base folder for all vectorstore data
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_BASE_PATH, exist_ok=True)  # Ensure the vectorstore directory exists

# Pre-initialize components (embeddings, vectorstore, retriever)
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/modernbert-embed-base", cache_folder="./saved_model")
llm_chat = ChatDeepSeek(model="deepseek-chat", temperature=0, api_key=DEEPSEEK_API_KEY, api_base='https://api.deepseek.com')
llm_resoner = ChatDeepSeek(model="deepseek-reasoner", temperature=0, api_key=DEEPSEEK_API_KEY, api_base='https://api.deepseek.com')

# Pydantic model
class AskQuestionRequest(BaseModel):
    question: str = Field(..., description="Question about the PDF.")
    pdf_name: str = Field(..., description="Name of the uploaded PDF.")

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
        # Read and print the raw body as a string
        body = await request.body()  # This reads the request body
        print(body.decode('utf-8'))  # Decode bytes to string
        
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
