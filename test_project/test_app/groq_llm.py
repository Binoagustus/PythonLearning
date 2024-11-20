import os
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm = ChatGroq(
    # model="llama-3.1-70b-versatile",
    # model = "llama-3.1-8b-instant", #20000 tokens/min
    model = "gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI assistant tasked with answering questions strictly based on the provided context. You must not generate any information or answer that is not explicitly present in the context. If the answer cannot be found in the context, respond only with "The answer is not available in the context." Do not provide additional explanations, guesses, or outside information.
            Context:
            {context}
            Question:
            {question}
            Answer:""",
        ),
        ("human", "{input}"),
    ]
)

def call_chain(context, user_input):
    chain = prompt | llm
    response = chain.invoke(
        {
            "context": context,
            "question": user_input,
            "input": user_input,
        }
    )
    return response.content