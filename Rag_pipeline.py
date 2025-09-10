import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


# 1. Load PDFs
def pdf_loader(pdf_folder="data/"):
    docs = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            docs.extend(loader.load())
    return docs


# 2. Split into chunks
def chunks_splitter(documents, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(documents)


# 3. Select Embeddings
def get_embeddings(provider="huggingface"):
    if provider == "google":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# 4. Create Vectorstore
def create_vectorstore(chunks, provider="huggingface", persist_dir="vectorstore"):
    embeddings = get_embeddings(provider)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_dir)
    return vectorstore


if __name__ == "__main__":
    docs = pdf_loader()
    print(f"Loaded {len(docs)} documents")
    
    chunks = chunks_splitter(docs)
    print(f"Split into {len(chunks)} chunks")
    
    vs = create_vectorstore(chunks, provider="huggingface")  # change to "google" if you want
    print("✅ Vectorstore created and saved!")
