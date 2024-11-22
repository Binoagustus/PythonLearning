from itertools import chain
import torch
from pgvector.psycopg2 import register_vector
from .db import get_connection
from .utils import get_query_embedding

from pgvector.psycopg2 import register_vector

template = """[INST]
You are a friendly documentation search bot.
Use following piece of context to answer the question.
If the context is empty, try your best to answer without it.
Never mention the context.
You must not generate any information or answer that is not explicitly present in the context.
If the answer cannot be found in the context, respond only with "The answer is not available in the context."
Do not provide additional explanations, guesses, or outside information
Try to keep your answers concise unless asked to provide details.

Context: {context}
Question: {question}
[/INST]
Answer:
"""

def get_retrieval_condition(query_embedding, threshold=0.6):
    # Convert query embedding to a string format for SQL query
    query_embedding_str = ",".join(map(str, query_embedding))

    # SQL condition for cosine similarity
    condition = f"(embedding <=> '[{query_embedding_str}]') < {threshold} ORDER BY embedding <=> '[{query_embedding_str}]'"
    return condition


def rag_query(query):
    # Generate query embedding
    query_embedding = get_query_embedding(query)

    # Retrieve relevant embeddings from the database
    retrieval_condition = get_retrieval_condition(query_embedding)

    conn = get_connection()
    register_vector(conn)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT content FROM pdf_chunks WHERE {retrieval_condition} LIMIT 5"
    )
    retrieved = cursor.fetchall()
    print(retrieved)
    return retrieved

# rag_query("project Manager")