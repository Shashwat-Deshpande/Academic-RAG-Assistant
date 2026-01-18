import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader


load_dotenv()

def start_ingestion():
    pdf_path = "data/ML_basics.pdf" 
    print(f"--- 1. Loading PDF: {pdf_path} ---")
    
    loader = DirectoryLoader("data/", glob="./*.pdf", loader_cls=PyMuPDFLoader)
    data = loader.load()
    print("--- 2. Splitting into Chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    
    print(f"--- 3. Creating Vectors for {len(chunks)} chunks (Please wait...) ---")
    # This uses a free HuggingFace model to turn text into math
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create the Vector Store
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    print("--- 4. Saving to Local Database ---")
    vector_db.save_local("faiss_index")
    print("--- SUCCESS: Vector Database Created and Saved! ---")

if __name__ == "__main__":
    start_ingestion()