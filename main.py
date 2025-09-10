import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 🔑 Absolute path for vectorstore
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")


# -----------------------------
# PDF Utilities
# -----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(VECTORSTORE_DIR)  # ✅ save with absolute path


# -----------------------------
# Conversational Chain
# -----------------------------
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, say:
    "Answer is not available in the context."
    Do not make up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# -----------------------------
# User Query Handling
# -----------------------------
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not os.path.exists(VECTORSTORE_DIR):
        st.warning("⚠️ No vectorstore found. Please upload and process a PDF first.")
        return

    new_db = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=3)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Show answer
    st.write("**Reply:** ", response["output_text"])

    # Show sources (page numbers if available)
    with st.expander("📄 Sources"):
        for doc in docs:
            st.markdown(f"- Page {doc.metadata.get('page', 'N/A')}")


# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config("Chat PDF")
    st.header("📑 Chat with PDF using Gemini")

    # 🔥 Preload vectorstore if exists
    if os.path.exists(VECTORSTORE_DIR):
        st.success("✅ Prebuilt financial vectorstore loaded!")
    else:
        st.warning("⚠️ No prebuilt vectorstore found. Please upload PDFs.")

    # Input box
    user_question = st.text_input("Ask a Question (Tesla/Apple financial reports preloaded)")

    if user_question:
        user_input(user_question)

    # Sidebar upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on Submit & Process",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Vectorstore updated with your PDFs!")


if __name__ == "__main__":
    main()
