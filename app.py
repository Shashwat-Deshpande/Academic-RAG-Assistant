import streamlit as st
import os, tempfile, pyttsx3
from dotenv import load_dotenv

# --- 1. Core LangChain & LLM ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 2. Vector Store & Document Loading ---
from langchain_community.vectorstores import FAISS
import pymupdf4llm  # Handles text and identifies tables/images
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 3. Advanced Retrieval Components ---
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_classic.chains import RetrievalQA

load_dotenv()

# ================================
# 🔊 LOCAL AUDIO FUNCTION (pyttsx3)
# ================================
def speak_text_local(text):
    """Speaks the answer using the local system voice driver."""
    try:
        # Clean text: remove markdown symbols so the AI doesn't say 'hash' or 'star'
        clean_text = text.replace("*", "").replace("#", "").replace("-", " ")
        
        # Initialize engine locally inside the function for stability
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)  # Natural speaking speed
        
        engine.say(clean_text)
        engine.runAndWait()
        
        # Release the driver so it's ready for the next question
        engine.stop()
    except Exception as e:
        st.error(f"Audio Error: {e}")

# ================================
# 🔍 HYBRID + RERANK RETRIEVER
# ================================
def get_advanced_retriever(chunks, embeddings):
    # Semantic Search
    faiss_retriever = FAISS.from_documents(chunks, embeddings).as_retriever(search_kwargs={"k": 5})
    
    # Keyword Search
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    
    # Combine both (Hybrid)
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.3, 0.7]
    )
    
    # Reranker: Filters the top results for accuracy
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)

# ================================
# 📄 DOCUMENT PROCESSING
# ================================
def process_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name

    # Extract text as Markdown (helps preserve chart/table context)
    extracted_text = pymupdf4llm.to_markdown(pdf_path)
    documents = [Document(page_content=extracted_text, metadata={"source": uploaded_file.name})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    retriever = get_advanced_retriever(chunks, embeddings)
    os.remove(pdf_path)
    return retriever

# ================================
# 🖥️ STREAMLIT UI
# ================================
st.set_page_config(page_title="Local Voice RAG Assistant", layout="wide")
st.title("🤖 Local Audio RAG: Text + Charts")

with st.sidebar:
    st.header("Document Center")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Build Advanced Index"):
        with st.spinner("Processing document and visuals..."):
            st.session_state.retriever = process_document(uploaded_file)
            st.success("Brain Ready!")

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ================================
# 🤖 CHAT LOGIC
# ================================
if prompt := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "retriever" in st.session_state:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=st.session_state.retriever,
            return_source_documents=True 
        )
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                # Get the response
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]
                
                # 1. Show the answer in UI
                st.markdown(answer)
                
                # 2. Trigger Local Speech (Answer only)
                speak_text_local(answer)
                
                # 3. Show Visual Evidence (Citations)
                if sources:
                    with st.expander("🔍 View Sources"):
                        for i, doc in enumerate(sources):
                            st.write(f"**Source {i+1}:**")
                            st.caption(doc.page_content[:300] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.warning("Please upload a PDF first!")