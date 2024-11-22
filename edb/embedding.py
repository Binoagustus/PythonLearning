# importing all the required modules
import PyPDF2
from PyPDF2 import PdfReader
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def generate_embeddings(text):
    return model.encode([text])[0].tolist()

def get_embeddings(texts):
    embeddings = []  # Initialize an empty list to store embeddings
    for text in texts:
        embedding = model.encode(text, convert_to_numpy=True)
        embeddings.append(embedding.tolist())  # Append each embedding
    return embeddings

def read_pdf_file(pdf_path):
    pdf_document = PyPDF2.PdfReader(pdf_path)
    lines = []
    for page_number in  range(len(pdf_document.pages)):
        page = pdf_document.pages[page_number]
        text = page.extract_text()
        lines.extend(text.splitlines())
    return lines

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks