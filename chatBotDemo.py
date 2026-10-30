
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Page Config
st.set_page_config(page_title="C++ RAG Chatbot")
st.title("C++ RAG Chatbot")
st.write("Ask any question related to C++ Introduction")

# Load and cache the vector database
@st.cache_resource
def load_vectorstore():
    # Load document
    loader = TextLoader("CPP.txt", encoding="utf-8")
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector database
    db = FAISS.from_documents(final_documents, embeddings)

    return db


db = load_vectorstore()

# User query
query = st.text_input("Enter your question about C++")

if query:
    docs = db.similarity_search(query, k=3)

    st.subheader("Retrieved Context:")

    for i, doc in enumerate(docs):
        st.markdown(f"**Result {i + 1}:**")
        st.write(doc.page_content)
