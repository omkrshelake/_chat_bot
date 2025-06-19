import streamlit as st
import pandas as pd
import torch
import os
from PyPDF2 import PdfReader
import docx
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# Streamlit config
st.set_page_config(page_title="📄 QA Chatbot", layout="wide")
st.title("🧠 AI Assistant")

# Load QA model once
@st.cache_resource
def load_model():
    model_name = "microsoft/xdoc-base-squad2.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# File extractors
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    return "\n".join([page.extract_text() or '' for page in reader.pages])

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="ignore")

def extract_text_from_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        df = df.dropna(how='all')
        docs = df.astype(str).agg(" ".join, axis=1).values.tolist()
        docs = [doc.strip() for doc in docs if doc.strip()]
        return docs
    except Exception as e:
        st.error(f"❌ Error processing CSV: {e}")
        return []

# TF-IDF functions
def build_tfidf_index(docs):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
    doc_vectors = vectorizer.fit_transform(docs)
    return vectorizer, doc_vectors

def retrieve_context(question, vectorizer, doc_vectors, docs, top_k=3):
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [docs[i] for i in top_indices]

# QA
def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
        return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

# Initialize LLM
llm = ChatOpenAI(temperature=0.5)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Always answer kindly and informatively."),
    ("human", "Context:\n{contexts}\n\nQuestion: {question}")
])

chain = prompt | llm

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload file
uploaded_file = st.file_uploader("Upload a file (CSV, PDF, DOCX, or TXT)", type=["csv", "pdf", "docx", "txt"])
question = st.chat_input("Ask a question")

if question:
    with st.spinner("Processing your question..."):
        context = ""

        if uploaded_file:
            ext = uploaded_file.name.split(".")[-1].lower()
            if ext == "csv":
                docs = extract_text_from_csv(uploaded_file)
            elif ext == "pdf":
                docs = [extract_text_from_pdf(uploaded_file)]
            elif ext == "docx":
                docs = [extract_text_from_docx(uploaded_file)]
            elif ext == "txt":
                docs = [extract_text_from_txt(uploaded_file)]
            else:
                st.error("Unsupported file type.")
                st.stop()

            if not docs:
                st.warning("No valid text content found.")
                st.stop()

            vectorizer, doc_vectors = build_tfidf_index(docs)
            contexts = retrieve_context(question, vectorizer, doc_vectors, docs, top_k=3)
            context = "\n\n".join(contexts)

        # Generate LLM answer
        
        ans = chain.invoke({"contexts": context, "question": question})

        # Store in chat history
        st.session_state.chat_history.append((question, ans.content))

# Display chat history
for user_msg, ai_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("ai"):
        st.markdown(ai_msg)

# If no question yet
if not st.session_state.chat_history:
    st.info("👋 Hello! I'm your AI assistant. You can ask me questions directly, or upload a document to get answers from it.")
