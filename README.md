# 🤖 Verified RAG Assistant with Automated Audit

An advanced Retrieval-Augmented Generation (RAG) system built to analyze technical academic documents with zero hallucinations.

## 📊 Performance Benchmarks
* **Faithfulness (Anti-Hallucination):** 1.0 (Verified via Ragas Framework)
* **Context Precision:** 0.90
* **Answer Relevancy:** 0.92

## 🛠️ Key Features
* **Structural Parsing:** Uses `PyMuPDF4LLM` to maintain context for LaTeX formulas and tables.
* **Hybrid Retrieval:** Combines FAISS (Semantic) and BM25 (Keyword) search.
* **Re-ranking:** Integrated `Flashrank` to optimize context precision before LLM inference.
* **Multi-Modal Output:** Real-time UI via Streamlit with integrated local Text-to-Speech (pyttsx3).

## ⚙️ Setup Instructions

Follow these steps to run the assistant locally on your machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/Shashwat-Deshpande/Academic-RAG-Assistant.git](https://github.com/Shashwat-Deshpande/Academic-RAG-Assistant.git)
cd Academic-RAG-Assistant

# Create the environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

pip install -r requirements.txt

# To launch the Streamlit interface
streamlit run app.py
