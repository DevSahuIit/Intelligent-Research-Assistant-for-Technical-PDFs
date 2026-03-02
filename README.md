# 📘 Intelligent Research Assistant for Technical PDFs
A production-grade, multi-user Retrieval-Augmented Generation (RAG) system for analyzing technical PDFs with citation-grounded responses and evaluation-driven validation.

This project goes beyond a basic chatbot. It implements cloud-hosted vector storage, namespace-based user isolation, multi-session conversational memory, retrieval tuning, and answer evaluation using RAGAS.

---

## 🚀 Features

- 🚀 Production-grade Retrieval-Augmented Generation (RAG) pipeline  
- 👥 Multi-user architecture with namespace-based cloud isolation (Pinecone)  
- 💬 Multi-session conversational memory with stateful history management  
- 🔁 History-aware query rewriting for follow-up question handling  
- 🎯 Tuned MMR retrieval for relevance–diversity balance  
- 🧠 Structure-aware document chunking to preserve semantic boundaries  
- 🔎 High-quality semantic search using sentence-transformers embeddings  
- ☁️ Persistent cloud vector storage with deduplication logic  
- 📑 Citation-grounded responses (source + page traceability)  
- 📊 RAGAS-based evaluation for faithfulness validation  
- 📈 LangSmith tracing for observability and debugging  

---

## 🏗 System Architecture

User  
→ Member ID  
→ Pinecone Namespace  
→ Multi-Session Memory  
→ History-Aware Query Rewriting  
→ MMR Retrieval  
→ Context Injection  
→ LLM Generation (Groq - LLaMA 3.1)  
→ Citation Rendering  
→ Optional RAGAS Evaluation  

---

## 🧠 Core Concepts Implemented

### 1. Retrieval-Augmented Generation (RAG)
Answers are generated strictly from retrieved document chunks, reducing hallucination and improving reliability.

### 2. Namespace-Based User Isolation
Each `member_id` maps to a Pinecone namespace, enabling:
- Persistent cloud storage  
- User-level document isolation  
- Restoration after restart  

### 3. Multi-Session Conversational Memory
Each user can create multiple chat sessions.  
Sessions maintain separate history while sharing the same document namespace.

### 4. Structure-Aware Chunking
Instead of naive text splitting:
- Detects numbered headings (1.1, 2.3, etc.)  
- Detects ALL-CAPS sections  
- Prevents cross-section semantic drift  

### 5. Duplicate Embedding Prevention
Chunk hashing (`content + source`) prevents repeated vector insertions when uploading the same document again.

### 6. Retrieval Engineering
- History-aware retriever (`create_history_aware_retriever`)  
- MMR search (`k=6`, tuned lambda)  
- Metadata preservation for citation traceability  

### 7. Hallucination Mitigation
Strict prompting enforces:
- Context-only answers  
- Explicit fallback when answer is not found  
- Mandatory citation awareness  

---

## 📊 Evaluation with RAGAS

The system integrates RAGAS to evaluate:

- **Faithfulness** – Measures whether the answer is grounded in retrieved context.

Evaluation is optional and can be toggled from the UI.

This adds measurable validation to the RAG pipeline.

---

## ⚠ Challenges Solved

### Pinecone Namespace Restoration
Resolved issues where previous uploads were not loading after restart by properly inspecting namespace stats.

### Duplicate Vector Growth
Prevented unnecessary vector growth using content-based hashing before insertion.

### Streamlit State Management
Implemented structured session storage using:


to prevent session mixing.

### RAGAS Compatibility with Groq
Groq does not support `n > 1`, causing evaluation failures.  
Solution: Use OpenAI API for evaluation while keeping Groq for generation.

### Hallucination Control
Refined system prompts to strictly enforce contextual grounding.

---

## 🧰 Tech Stack

- LangChain  
- Pinecone (Vector Database)  
- Groq (LLaMA 3.1)  
- HuggingFace sentence-transformers  
- RAGAS  
- Streamlit  
- LangSmith  

---

## 📦 Installation

```bash
pip install -r requirements.txt

PINECONE_API_KEY=
GROQ=
OPENAI_API_KEY=   
LANGCHAIN_API_KEY=
streamlit run app.py ## too run the app or use the live link