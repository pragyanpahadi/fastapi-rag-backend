# RAG Engine API

An asynchronous conversational RAG backend engine built with FastAPI. It implements a clean, decoupled architecture for document processing, metadata tracking, vector space embeddings, and rolling session based chat memory.

## Key Architectural Features

- **Custom RAG Architecture:** Engineered from scratch using native client abstractions (`qdrant-client`, `openai`, `redis`) without high level wrapper frameworks (eg, LangChain,LlamaIndex) to maximize execution control and optimize token efficiency.
- **Dual Chunking Processing:** Supports both deterministic token based chunking (`tiktoken`) and structural recursive character chunking algorithms.
- **Local Embedding Vectorization:** Utilizes `sentence-transformers` (`all-MiniLM-L6-v2`) executed locally on the CPU, outputting dense 384-dimensional mathematical arrays.
- **Decoupled Hybrid Storage:** - **PostgreSQL:** Manages persistent relational schemas for document indexing and chunk metadata.
  - **Qdrant:** High performance vector database utilized for geometric cosine-similarity searches.
- **Stateful Session Memory:** Uses Redis as an ephemeral cache layer to store the last 10 conversational turns per `session_id`, preserving contextual continuity while eliminating context window bloat.
- **Structured Intent Extraction:** Enforces deterministic JSON outputs from the LLM via native JSON-mode schema parsing to seamlessly detect and extract interview booking metadata (Name, Email, Date, Time).
- **Graceful Offline Fallback:** Includes a robust evaluation fallback configuration. If no OpenAI API key is detected, the service automatically diverts to a mock deterministic inference matrix, preventing runtime failures during localized testing.

## Infrastructure Topology

```text
                  ┌────────────────────────┐
                  │    FastAPI Gateway     │
                  └───────────┬────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
   ┌───────────────────┐             ┌───────────────────┐
   │  Ingestion Route  │             │    Chat Route     │
   └─────────┬─────────┘             └─────────┬─────────┘
             │                                 │
     [Extract & Chunk]                         ▼
             │                       ┌───────────────────┐
     ┌───────┼───────┐               │   Redis Memory    │
     ▼       ▼       ▼               └─────────┬─────────┘
┌────────┐┌──────┐┌────────┐                   │ [Fetch History]
│Postgres││Local ││ Qdrant │                   ▼
│(Meta)  ││Embed ││(Vector)│         ┌───────────────────┐
└────────┘└──────┘└────────┘         │    RAG Engine     │
                                     └─────────┬─────────┘
                                               │ [Vector Search]
                                               ▼
                                     ┌───────────────────┐
                                     │    OpenAI API     │
                                     └───────────────────┘
```

## Quick Start & Environment Setup

### 1. Provision Infrastructure Dependencies
Start up decoupled operational infrastructure layers (PostgreSQL, Qdrant, and Redis) using Docker Compose:
```bash
docker-compose up -d
```

### 2. Configure Virtual Environment & Dependencies
Initialize a clean Python 3.10+ isolated workspace and resolve all required core dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configurations
The system reads runtime variables from your machine's environment space. Create a `.env` file or export variables directly:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_db
OPENAI_API_KEY=your-actual-api-key-here
```
*Note: If `OPENAI_API_KEY` is omitted, the system defaults to `mock-key` and triggers the local fallback matrix so the API routes remain fully testable without network dependencies.*

### 4. Boot the Application Server
Start up the asynchronous application runtime using Uvicorn:
```bash
uvicorn app.main:app --reload
```
The API documentation gateway will initialize instantly and can be evaluated visually at: **http://localhost:8000/docs**

## Core API Specification Reference

### 1. Document Ingestion Pipeline
Processes binary unstructured uploads, segments strings into vectors, persists transactional tracking blocks into Postgres, and vectors into Qdrant.

- **Endpoint:** `POST /api/v1/upload`
- **Content-Type:** `multipart/form-data`
- **Payload Parameters:**
  - `file`: Binary Data (PDF, TXT)
  - `strategy`: String (`token` or `recursive`)

### 2. Conversational RAG Gateway
Handles rolling chat history retrievals, issues top-k matrix similarity evaluations to Qdrant, updates context windows, and structural intent classifications.

- **Endpoint:** `POST /api/v1/chat`
- **Content-Type:** `application/json`
- **Payload Schema:**
```json
{
  "session_id": "session_user_9921",
  "message": "I would love to book my interview for tomorrow at 2 PM. My name is Pragyan and my email is prag@test.com"
}
```