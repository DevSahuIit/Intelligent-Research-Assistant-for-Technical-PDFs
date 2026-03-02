from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langsmith import Client
import streamlit as st
import re
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Advanced-bot"

## setting up the hugging face embeddings
os.environ['HF_TOKEN'] = os.getenv("HUGGINFACE_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
api_key = os.getenv("GROQ")


## setting up the langsmith and checking if it works 
client = Client()



st.title("Intelligent-Research-Assistant-for-Technical-PDFs")
st.write("LangSmith connected successfully")
st.write("Upload The Pdfs")

llm = ChatGroq(model ="llama-3.1-8b-instant" ,groq_api_key=api_key)


#chat interface - making sessions

## statefully manage Chat history
if 'store' not in st.session_state:
    st.session_state.store = {}


## for uploading the files
all_docs = []

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# =========================
# SESSION MANAGER (Sidebar)
# =========================

if "store" not in st.session_state:
    st.session_state.store = {}

# Existing sessions list
existing_sessions = list(st.session_state.store.keys())

# Dropdown to select session
selected_session = st.sidebar.selectbox(
    "Select Session",
    options=existing_sessions if existing_sessions else ["default-session"]
)

# Input to create new session
new_session = st.sidebar.text_input("Create New Session")

if st.sidebar.button("Add Session"):
    if new_session and new_session not in st.session_state.store:
        st.session_state.store[new_session] = ChatMessageHistory()
        st.sidebar.success(f"Session '{new_session}' created!")
        selected_session = new_session

# Final session_id used
session_id = selected_session



## separating all files and saving it on local temp 
os.makedirs("temp", exist_ok=True)
if uploaded_files:
    for uploaded_file in uploaded_files:   # From Streamlit uploader
        file_path = f"./temp/{uploaded_file.name}"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        for doc in docs:
            doc.metadata = {
                "source": uploaded_file.name,        # Clean filename
                "page": doc.metadata.get("page", 0) + 1  # Human-friendly page
            }
        
        all_docs.extend(docs)

## chunking the data we have
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=800,
#     chunk_overlap=150
# )


## chunk the data for 
def structure_aware_chunking(documents, chunk_size=800, chunk_overlap=100):
    structured_docs = []
    
    for doc in documents:
        text = doc.page_content
        
        # Detect headings:
        # - Numbered sections like 1., 1.1, 2.3.4
        # - ALL CAPS headings
        sections = re.split(
        r"(?=\n\d+(?:\.\d+)*\s)|(?=\n[A-Z][A-Z\s]{3,}\n)",
        text
        )
        
        for section in sections:
            if not isinstance(section, str):
                continue

            section = section.strip()
            if not section:
                continue
            
            # Now chunk inside section without mixing
            start = 0
            while start < len(section):
                end = start + chunk_size
                chunk_text = section[start:end]
                
                structured_docs.append(
                    Document(
                        page_content=chunk_text.strip(),
                        metadata=doc.metadata
                    )
                )
                
                start += chunk_size - chunk_overlap
    
    return structured_docs

if all_docs:

    split_docs = structure_aware_chunking(all_docs, chunk_size=800)

    if not split_docs:
        st.error("No valid text found in PDFs.")
        st.stop()

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever()

else:
    st.warning("Please upload at least one PDF.")
    st.stop()


vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever()

rewrite_system_prompt = """
You are a query rewriting assistant.

Given the chat history and the latest user question,
rewrite the question into a clear standalone question.

Do NOT answer the question.
Only return the rewritten standalone question.
"""

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rewrite_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    rewrite_prompt
)


qa_system_prompt = """
You are an AI research assistant helping users analyze academic documents.

Your task is to answer the user's question using ONLY the information provided inside the <context> tags.

<context>
{context}
</context>

Instructions:

1. If the answer can be reasonably inferred from the provided context, provide a clear and academically accurate answer.
2. Do NOT use external knowledge.
3. Do NOT fabricate information.
4. If the context truly does not contain relevant information, respond exactly with:
   "The answer is not found in the provided documents."
5. When possible, mention the source and page number.

Be precise, structured, and factual.
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

user_input = st.text_input("Your Question:")

if user_input:

    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": session_id}
        }
    )

    
    st.write("### Assistant Response")
    st.write(response["answer"])


    st.write("Retrieved Documents:")
    
    if "context" in response:

        st.write("### Sources: ")

        seen = set()

        for doc in response["context"]:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")

            citation = f"{source} (Page {page})"

            # Avoid duplicates
            if citation not in seen:
                st.write("-", citation)
                seen.add(citation)



