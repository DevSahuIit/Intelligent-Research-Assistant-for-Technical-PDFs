from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langsmith import Client
from datasets import Dataset
from ragas import evaluate
import streamlit as st
import hashlib
import re
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Advanced-bot"

## setting up the hugging face embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
## setting up the pinecone 
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

## creating a function to get the uploaded file of the user with his user id
def get_uploaded_files(pc, index_name, namespace):
    index = pc.Index(index_name)
    stats = index.describe_index_stats()

    # If namespace does not exist
    if namespace not in stats["namespaces"]:
        return []

    # If namespace exists but no vectors
    if stats["namespaces"][namespace]["vector_count"] == 0:
        return []

    # Fetch small batch of vectors properly
    query_response = index.query(
        namespace=namespace,
        vector=[0.1] * 384,   # non-zero dummy vector
        top_k=50,
        include_metadata=True
    )

    files = set()

    for match in query_response.get("matches", []):
        metadata = match.get("metadata", {})
        if "source" in metadata:
            files.add(metadata["source"])

    return list(files)

index_name = "research-assistant"

# Create index if not exists
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
api_key = os.getenv("GROQ")


## setting up the langsmith and checking if it works 
client = Client()



st.title("Intelligent-Research-Assistant-for-Technical-PDFs")
st.write("LangSmith connected successfully")

llm = ChatGroq(model ="llama-3.1-8b-instant" ,groq_api_key=api_key)



## for uploading the files
all_docs = []

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

member_id = st.sidebar.text_input("Enter Member ID")
namespace = member_id


if member_id:
    uploaded_files_list = get_uploaded_files(pc, index_name, namespace)
    
    if uploaded_files_list:
        st.sidebar.write("## Uploaded Files Earlier:")
        for file in uploaded_files_list:
            st.sidebar.write("-", file)
    else:
        st.sidebar.write("No documents uploaded before.")

else :
    st.warning("Please enter your Member ID.")
    st.stop()



# =========================
# MULTI-SESSION PER USER
# =========================

if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}

if member_id not in st.session_state.chat_store:
    st.session_state.chat_store[member_id] = {}

user_sessions = st.session_state.chat_store[member_id]

new_session = st.sidebar.text_input("Create New Session")

if st.sidebar.button("Add Session"):
    if new_session and new_session not in user_sessions:
        user_sessions[new_session] = ChatMessageHistory()
        st.sidebar.success(f"Session '{new_session}' created!")
        st.rerun()

if not user_sessions:
    user_sessions["default-session"] = ChatMessageHistory()

session_id = st.sidebar.selectbox(
    "Select Session",
    options=list(user_sessions.keys())
)

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

# PINECONE VECTORSTORE LOGIC
if uploaded_files:

    split_docs = structure_aware_chunking(all_docs, chunk_size=800)

    if not split_docs:
        st.error("No valid text found in PDFs.")
        st.stop()

    docs_with_ids = []
    ids = []

    for doc in split_docs:
        unique_string = doc.page_content + doc.metadata["source"]
        content_hash = hashlib.md5(unique_string.encode()).hexdigest()
        docs_with_ids.append(doc)
        ids.append(content_hash)

    vectorstore = PineconeVectorStore.from_documents(
        documents=docs_with_ids,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace,
        ids=ids
    )

else:
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )

    # namespace existence check
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        if namespace not in stats["namespaces"]:
            st.warning("No documents found for this Member ID. Please upload PDFs first.")
    except Exception:
        st.warning("Could not retrieve namespace stats.")


## making a cache resource 
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "lambda_mult": 0.5}
)


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
    return st.session_state.chat_store[member_id][session_id]

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
    config={"configurable": {"session_id": session_id}}
)

    # -----------------------------
    # Extract RAG components
    # -----------------------------
    question = user_input
    answer = response["answer"]
    contexts = [doc.page_content for doc in response.get("context", [])]

    # Display answer
    st.write("### Assistant Response")
    st.write(answer)

    # -----------------------------
    # Optional RAGAS Evaluation
    # -----------------------------
    if st.sidebar.checkbox("Run RAGAS Evaluation"):

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        dataset = Dataset.from_dict(data)

        evaluator_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0,
        n=1
            )
        result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm
        )

        st.write("### RAGAS Evaluation Scores")

        scores = result.scores[0]
        st.write(
        f"Faithfulness: {round(scores["faithfulness"], 3)}",
        )

    
    st.write("### Assistant Response")
    st.write(response["answer"])


    
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



