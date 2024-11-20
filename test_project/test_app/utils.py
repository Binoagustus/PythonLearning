import os
import numpy as np
from numpy.linalg import norm
import psycopg2
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

#text extractor
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

#text to chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#create embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def get_embeddings(texts):
    embeddings = []  # Initialize an empty list to store embeddings
    for text in texts:
        embedding = model.encode(text, convert_to_numpy=True)
        embeddings.append(embedding.tolist())  # Append each embedding
    return embeddings

load_dotenv()
DATABASE_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST', 'localhost'),  # Default to localhost if not set
    'port': os.getenv('DB_PORT', '5432')
}
conn = psycopg2.connect(**DATABASE_CONFIG)
cur = conn.cursor()
def create_pdf():
    cur.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE IF NOT EXISTS pdf_chunks (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding VECTOR(384)
    );
    """)
    print("Table is created and indexed")
    conn.commit()

def create_index_if_not_exists():
    # Check if the index already exists
    cur.execute("""
        SELECT 1
        FROM pg_indexes
        WHERE tablename = 'pdf_chunks' AND indexname = 'idx_embedding_cosine';
    """)
    exists = cur.fetchone()

    # Create the index if it doesn't exist
    if not exists:
        cur.execute("""
            CREATE INDEX idx_embedding_cosine
            ON pdf_chunks USING ivfflat (embedding) WITH (lists = 100);
        """)
        conn.commit()

def insert_embedding(text_chunks, embedding_list):
    with conn.cursor() as cursor:
        # Make sure text_chunks and embedding_list are correct 1-D arrays
        for text_chunk, embedding in zip(text_chunks, embedding_list):
            # Ensure embedding is a 1-D array or list (e.g., [0.1, 0.2, 0.3, ...])
            cursor.execute("""
                INSERT INTO pdf_chunks (content, embedding) 
                VALUES (%s, %s::vector);
            """, (text_chunk, embedding))  # embedding should be 1-D
        conn.commit()

def similarity_search(query_embedding):
    """Similarity search with user query"""
    with conn.cursor() as cursor:
        cursor.execute("""
        SELECT content, embedding <-> %s::vector AS similarity
        FROM pdf_chunks
        """, (query_embedding,))
        results = cursor.fetchall()
        return results
      
def get_query_embedding(input_string):
    """Query embedding for user input"""  
    return model.encode([input_string])[0].tolist()
