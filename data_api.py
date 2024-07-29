from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
# from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from langchain_ollama import OllamaEmbeddings

# Load environment variables
load_dotenv()

# Define paths and configurations
index_path = "faiss_index"
temp_dir = "tempDir"

# Initialize FastAPI
app = FastAPI()

# Initialize LLM
llm = ChatOllama(
    model="llama3",
    temperature=0,
)

# Define prompt template
prompt_template = """Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Questions: {input}
"""
prompt = ChatPromptTemplate.from_template(template=prompt_template)

# Embeddings configuration
embeddings = OllamaEmbeddings(model="llama3")

def create_vector_store(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(pages)

    vectorstore = FAISS.from_documents(all_splits, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def load_vector_store():
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def get_response(query, retriever):
    chain = ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=True)
    chat_history = []
    result = chain({"question": query, "chat_history": chat_history})
    return result['answer']

# Define request model for query endpoint
class QueryRequest(BaseModel):
    query: str

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    vectorstore = create_vector_store(file_path)
    return {"message": "Vector store created and saved."}

@app.post("/query/")
async def answer_query(request: QueryRequest):
    if not os.path.exists(index_path):
        raise HTTPException(status_code=400, detail="Vector store does not exist. Please upload a PDF first.")

    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    answer = get_response(request.query, retriever)
    return {"answer": answer}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Data Reader API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
