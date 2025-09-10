📑 Finance Chatbot with Google Gemini

An intelligent domain-specific chatbot built to query financial reports (10-K, 10-Q, etc.) for companies like Apple and Tesla.
The chatbot uses Google Gemini Pro (LLM) with RAG (Retrieval-Augmented Generation) to deliver accurate, contextual, and explainable answers from PDFs.

🚀 Features

Chat with PDFs → Upload any company’s annual/quarterly report and ask questions.

Preloaded Financial Data → Already supports Apple & Tesla 10-K filings.

Google Gemini Pro (LLM) → Provides detailed, natural language answers.

Vector Search (FAISS) → Efficient document retrieval with embeddings.

Daily Quota Handling → Uses Gemini’s free quota, resets every day.

Streamlit Frontend → Simple, interactive web UI.

Citations → Retrieves and displays source page numbers for transparency.

🛠️ Tech Stack

LLM → Google Gemini Pro
 (via langchain-google-genai)

Embeddings → GoogleGenerativeAIEmbeddings (embedding-001)

Vector Store → FAISS

Frontend → Streamlit

Document Parsing → PyPDF2, LangChain text splitter

⚡ Attempts with Hugging Face

We also tried replacing Gemini with Hugging Face models (FLAN-T5, etc.) for a free offline setup.

While technically functional, the results were very poor compared to Gemini:

Struggled with long, structured financial text.

Answers were incomplete and lacked reasoning.

Hence, the final project uses Gemini Pro as the main LLM.

📂 Project Structure
Finance-Chatbot/
│
├── app/
│   ├── rag_pipeline.py   # Loads PDFs, splits into chunks, builds vectorstore
│   ├── rag_qa.py         # Retrieval + QA pipeline with Gemini/HF
│   ├── main.py           # Streamlit UI (final app)
│
├── faiss_index/          # Prebuilt vectorstore (Apple + Tesla reports)
├── data/                 # Raw PDF reports
├── requirements.txt
├── README.md

▶️ Usage
1. Clone the repo
git clone https://github.com/your-username/finance-chatbot.git
cd finance-chatbot

2. Install dependencies
pip install -r requirements.txt

3. Set up API key

Create a .env file in the root with:

GOOGLE_API_KEY=your_google_gemini_key

4. Run Streamlit app
streamlit run app/main.py

5. Ask Questions 🎯

Example: “What was Apple’s net income in 2023?”

Example: “Summarize Tesla’s revenue breakdown.”

🌟 Example Queries

✅ “What was Apple’s net income in 2023?”
→ $96,995 million (from Apple 10-K, page 31).

✅ “Compare Tesla’s revenue growth over the last 3 years.”
→ Detailed answer with citations to Tesla 10-K.

🔮 Future Improvements

Add support for multi-model fallback (Gemini → GPT-4 → Hugging Face).

Enable highlighted text snippets for each answer.

Integrate SQL financial databases alongside PDFs.

📌 Conclusion

This project demonstrates:

Building a real-world RAG pipeline with Gemini.

Challenges of open-source LLMs vs. proprietary APIs.

A working end-to-end application with clear use-case: financial analysis.