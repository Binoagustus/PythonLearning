import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Hardcoded API key
api_key = 'api key from google gemini'

# Set the API key in the environment variables
os.environ['GOOGLE_API_KEY'] = api_key

# Configure the generative AI client with the API key
genai.configure(api_key=api_key)

def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversional_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not available in the context, just say, "answer is not available in the context", don't provide the wrong answer.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, processed_pdf_text):
    embeddings = GoogleGenerativeAIEmbeddings(api_key=api_key, model="models/embedding-001")
    text_chunks = get_text_chunks(processed_pdf_text)
    documents = [Document(page_content=chunk) for chunk in text_chunks]

    chain = get_conversional_chain()

    # Combine user question and processed PDF text as context
    context = f"{processed_pdf_text}\n\nQuestion: {user_question}"

    response = chain.invoke({"input_documents": documents, "question": user_question, "context": context})

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat With Multiple PDF", layout="wide")
    st.header("Chat with PDF's powered by Gemini 🙋‍♂️")

    with st.sidebar:
        st.title("Menu:")
        pdf_path_input = st.text_area("Enter the local paths to your PDF file(s), separated by commas", value="D:/Vigneshwaran'sData/PDF/Vigneshwaran_Resume.pdf")
        pdf_paths = [path.strip() for path in pdf_path_input.split(",")]

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        processed_pdf_text = get_pdf_text(pdf_paths)
        user_input(user_question, processed_pdf_text)

if __name__ == "__main__":
    main()