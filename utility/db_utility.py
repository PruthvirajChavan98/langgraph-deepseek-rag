import sqlite3

# Constants
DB_PATH = "pipelines.db"

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