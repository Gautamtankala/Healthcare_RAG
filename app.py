from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# ----------------------------------
# Load Healthcare Documents
# ----------------------------------

loader = TextLoader(
    "data/healthcare.txt",
    encoding="utf-8"
)

documents = loader.load()

print("Documents Loaded Successfully")
print(documents)

# ----------------------------------
# Split Documents
# ----------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

print(f"Total Chunks Created: {len(docs)}")

# ----------------------------------
# Create Embeddings
# ----------------------------------

embeddings = OpenAIEmbeddings()

# ----------------------------------
# Create Vector Database
# ----------------------------------

vector_db = FAISS.from_documents(
    docs,
    embeddings
)

print("FAISS Vector DB Created")

# ----------------------------------
# Load LLM
# ----------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ----------------------------------
# FastAPI Application
# ----------------------------------

app = FastAPI(
    title="Healthcare RAG API"
)

@app.get("/")
def home():
    return {
        "message": "Healthcare RAG API Running"
    }


@app.get("/ask")
def ask_question(query: str):

    retrieved_docs = vector_db.similarity_search(
        query,
        k=2
    )

    context = "\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = f"""
You are a healthcare assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return {
        "question": query,
        "answer": response.content,
        "context": context
    }