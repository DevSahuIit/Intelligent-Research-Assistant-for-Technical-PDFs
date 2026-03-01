from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
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
st.write("LangSmith connected successfully")


st.title("Intelligent-Research-Assistant-for-Technical-PDFs")
st.write("Upload The Pdfs")

llm = ChatGroq(model ="llama-3.1-8b-instant" ,groq_api_key=api_key)


## for uploading the files
all_docs = []

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)
## separating all files and saving it on local temp 

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
            r'(?=\n\d+(\.\d+)*\s)|(?=\n[A-Z][A-Z\s]{3,}\n)',
            text
        )
        
        for section in sections:
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

split_docs = structure_aware_chunking(all_docs, chunk_size=800)


vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
