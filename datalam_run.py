# import streamlit as st
# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import ConversationalRetrievalChain
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain_groq import ChatGroq
# from langchain_ollama import ChatOllama
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import SKLearnVectorStore

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# # Define paths and configurations
# index_path = "faiss_index"
# temp_dir = "tempDir"

# # Initialize LLM
# llm = ChatOllama(
#     #model="llama3-groq-tool-use",
#     model="llama3",
#     temperature=0,
# )

# # Define prompt template
# prompt_template = """Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question.
# <context>
# {context}
# <context>
# Questions:{input}
# """

# prompt = ChatPromptTemplate.from_template(template=prompt_template)

# # Embeddings configuration
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name="BAAI/bge-small-en-v1.5",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )

# def create_vector_store(file_path):
#     loader = PyPDFLoader(file_path)
#     pages = loader.load_and_split()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     all_splits = text_splitter.split_documents(pages)

#     vectorstore = FAISS.from_documents(all_splits, embeddings)
#     vectorstore.save_local(index_path)
#     return vectorstore

# def load_vector_store():
#     return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# def get_response(query, retriever):
#     chain = ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=True)
#     chat_history = []
#     result = chain({"question": query, "chat_history": chat_history})
#     return result['answer']

# def main():
#     st.title("Data Reader")

#     # Ensure the tempDir directory exists
#     if not os.path.exists(temp_dir):
#         os.makedirs(temp_dir)

#     uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

#     if uploaded_file:
#         file_path = os.path.join(temp_dir, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         st.session_state['uploaded_file_path'] = file_path
#         st.write("PDF uploaded successfully. Click the button below to create the vector store.")

#     if 'uploaded_file_path' in st.session_state:
#         if st.button("Create Vector Store"):
#             vectorstore = create_vector_store(st.session_state['uploaded_file_path'])
#             st.session_state['vectorstore_created'] = True
#             st.write("Vector store created and saved.")

#     if 'vectorstore_created' in st.session_state and st.session_state['vectorstore_created']:
#         vectorstore = load_vector_store()
#         retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

#         query = st.text_input("Enter your query:")
#         if st.button("Get Answer"):
#             if query:
#                 answer = get_response(query, retriever)
#                 st.write("Bot's Response:")
#                 st.write(answer)
#             else:
#                 st.write("Please enter a query.")
#     else:
#         st.write("Please upload a PDF file and create the vector store first.")

# if __name__ == "__main__":
#     main()
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_ollama import ChatOllama

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define paths and configurations
index_path = "faiss_index"
temp_dir = "tempDir"

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
<context>
Questions:{input}
"""

prompt = ChatPromptTemplate.from_template(template=prompt_template)

# Embeddings configuration
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

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

def main():
    st.title("Data Reader")

    # Ensure the tempDir directory exists
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    file_path = st.text_input("Enter the path to your PDF file:")

    if file_path:
        if st.button("Create Vector Store"):
            if os.path.exists(file_path):
                vectorstore = create_vector_store(file_path)
                st.session_state['vectorstore_created'] = True
                st.write("Vector store created and saved.")
            else:
                st.error("The provided file path does not exist.")

    if 'vectorstore_created' in st.session_state and st.session_state['vectorstore_created']:
        vectorstore = load_vector_store()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

        query = st.text_input("Enter your query:")
        if st.button("Get Answer"):
            if query:
                answer = get_response(query, retriever)
                st.write("Bot's Response:")
                st.write(answer)
            else:
                st.write("Please enter a query.")
    else:
        st.write("Please provide a valid file path and create the vector store first.")

if __name__ == "__main__":
    main()
