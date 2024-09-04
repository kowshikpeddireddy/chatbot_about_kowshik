import os

from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain


# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db")

# Define the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)


# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}

    """

)

document_chain=create_stuff_documents_chain(llm,prompt)
retrieval_chain=create_retrieval_chain(retriever,document_chain)

response=retrieval_chain.invoke({'input':"what does his dad do?"})
print(response['answer'])






