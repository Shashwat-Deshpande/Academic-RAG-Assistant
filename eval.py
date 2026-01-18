import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate

from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

from tabulate import tabulate

# 1. SETUP ENVIRONMENT
load_dotenv()

# 2. INITIALIZE COMPONENTS
print("--- 🔄 Initializing Evaluation Components ---")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# Load your existing local index
if not os.path.exists("faiss_index"):
    print("❌ Error: 'faiss_index' folder not found. Please run ingest.py first!")
    exit()

vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 3. DEFINE A TEST SET (Golden Dataset)
test_queries = [
    {
        "question": "What is the core definition of Empirical Risk Minimization?",
        "ground_truth": "Empirical Risk Minimization (ERM) is a principle in statistical learning theory which defines a family of learning algorithms and is used to give theoretical bounds on their performance."
    },
    {
        "question": "Who is the author of this document and what is their role?",
        "ground_truth": "The author is P. Murali, an Assistant Professor in the CSE Department at Aditya Engineering College."
    }
]

# 4. RUN INFERENCE
print("--- 🧪 Running Test Queries ---")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(),
    return_source_documents=True
)

eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

for item in test_queries:
    response = qa_chain.invoke({"query": item["question"]})
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(response["result"])
    eval_data["contexts"].append([doc.page_content for doc in response["source_documents"]])
    eval_data["ground_truth"].append(item["ground_truth"])

# 5. CALCULATE RAGAS SCORES
print("--- 📊 Calculating RAGAS Metrics ---")
dataset = Dataset.from_dict(eval_data)

# Wrap your LLM and Embeddings for Ragas
ragas_llm = LangchainLLMWrapper(llm)
ragas_emb = LangchainEmbeddingsWrapper(embeddings)

# Initialize metrics with the LLM
metrics = [
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    ContextPrecision(llm=ragas_llm),
    ContextRecall(llm=ragas_llm)
]

result = evaluate(
    dataset,
    metrics=metrics
)

# 6. OUTPUT & SAVE RESULTS
df = result.to_pandas()
df.to_csv("rag_evaluation_report.csv", index=False)

print("\n--- ✅ EVALUATION COMPLETE ---")
print(df[['question', 'faithfulness', 'answer_relevancy']])

print("\n" + "="*50)
print("             RAG EVALUATION RESULTS             ")
print("="*50)

# Calculate averages
avg_scores = df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean().to_frame().T
avg_scores.index = ['Average']

# Print the pretty table
print(tabulate(df[['user_input', 'faithfulness', 'answer_relevancy']], headers='keys', tablefmt='psql'))
print("\nSUMMARY METRICS:")
print(tabulate(avg_scores, headers='keys', tablefmt='psql', showindex=False))