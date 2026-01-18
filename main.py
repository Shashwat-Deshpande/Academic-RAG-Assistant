import os
import pyttsx3
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 1. Initialize the Voice Engine (SAPI5 is the Windows driver)
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id) # 1 is Female, 0 is Male
engine.setProperty('rate', 170)           # Adjust talking speed

def speak_text(text):
    """Function to make the PC speak"""
    engine.say(text)
    engine.runAndWait()

# 2. Setup LLM & Embeddings
print("--- Connecting to Memory ---")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the local database you created with ingest.py
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Use the updated Llama 3.1 model
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 3. The "Anti-Hallucination" Instruction
template = """You are a focused Academic Assistant. Use ONLY the following pieces of context to answer the question.
If the answer is not in the context, strictly say: "I'm sorry, that information is not in the document."
Do not use your own outside knowledge.

Context: {context}
Question: {question}
Helpful Answer:"""

QA_PROMPT = PromptTemplate.from_template(template)

# 4. Create the Retrieval Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(),
    chain_type_kwargs={"prompt": QA_PROMPT}
)

def start_chat():
    print("\n✅ AI is ready and Voice is enabled!")
    print("Ask anything about your PDF (type 'exit' to quit):")

    while True:
        query = input("\nYour Question: ")
        
        if query.lower() == 'exit':
            print("Exiting... Goodbye!")
            speak_text("Goodbye! Have a great day.")
            break

        print("Thinking...")
        
        # Get response from AI
        response = qa_chain.invoke(query)
        answer = response['result']
        
        print(f"\nAI Answer: {answer}")
        
        # Trigger the PC speakers
        speak_text(answer)

if __name__ == "__main__":
    start_chat()