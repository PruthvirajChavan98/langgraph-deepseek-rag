# RAG Application with LangGraph and DeepSeek Models

This repository contains a Python-based web application that integrates **FastAPI** for the backend, **Streamlit** for the frontend, and combines **LangGraph** with **ModernBERT** and **DeepSeek** models. The backend utilizes state-of-the-art NLP models for embedding, generation, and structured output.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Running the FastAPI app](#running-the-fastapi-app)
  - [Running the Streamlit app](#running-the-streamlit-app)
- [Model Information](#model-information)
  - [Embedding Model: ModernBERT](#embedding-model-modernbert)
  - [DeepSeek Models](#deepseek-models)
    - [DeepSeek-R1: Reasoning Model](#deepseek-r1-reasoning-model)
- [License](#license)

## Installation


### Steps

1. **Clone the repository:**
    ```bash
    git clone https://github.com/PruthvirajChavan98/langgraph-deepseek-rag.git
    cd langgraph-deepseek-rag
    ```

2. **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the FastAPI app

To start the FastAPI backend server, use the following command:

```bash
uvicorn main:app --reload
```

- The FastAPI app will be accessible at `http://127.0.0.1:8000`.
- The API documentation can be accessed via:
  - **Swagger UI**: `http://127.0.0.1:8000/docs`
  - **ReDoc UI**: `http://127.0.0.1:8000/redoc`

### Running the Streamlit app

The Streamlit frontend is located in the parent directory. To start the Streamlit app, run the following:

```bash
streamlit run app.py
```

- The app will be accessible at `http://localhost:8501`.

## Model Information

### Embedding Model: ModernBERT

This app uses the **ModernBERT** model from Huggingface as an embedding model for semantic understanding and retrieval. The model is loaded using **LangChain**:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/modernbert-embed-base", cache_folder="./saved_model")
```

- The `ModernBERT` model is optimized for embedding-based retrieval tasks and is cached locally for faster usage.

### DeepSeek Models

- **DeepSeek-Chat**: This model is used for generating structured output, which allows the system to provide organized, contextual responses based on the input.
  
- **DeepSeek-R1: Reasoning Model**: The **DeepSeek R1** model is responsible for complex reasoning tasks, allowing the system to handle reasoning, logical deductions, and multi-step problem-solving. This model is used for higher-order cognitive tasks and is integrated for providing reasoned outputs based on inputs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  
