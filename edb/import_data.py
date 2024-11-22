import numpy as np

from db import get_connection
from embedding import generate_embeddings, read_pdf_file, get_embeddings, get_pdf_text, get_text_chunks


def import_data(pdf_path):
    # data = read_pdf_file(pdf_path)
    data = get_pdf_text(pdf_path)
    chunks = get_text_chunks(data)

    embeddings = get_embeddings(chunks)

    conn = get_connection()
    cursor = conn.cursor()


    for doc_fragment, embedding in zip(chunks, embeddings):
            # Ensure embedding is a 1-D array or list (e.g., [0.1, 0.2, 0.3, ...])
            cursor.execute("""
                INSERT INTO embeddings (doc_fragment, embeddings) 
                VALUES (%s, %s::vector);
            """, (doc_fragment, embedding))  # embedding should be 1-D
    conn.commit()

    print(
        "import-data command executed. Data source:"
    )

import_data("D:\\Projects\\sample_data\\pdf\\Resume10.pdf")