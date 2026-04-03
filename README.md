# ResolveIQ — AI-Powered ServiceNow Resolution Engine

An intelligent incident and change request resolution engine built with **LangGraph**, **Gemini**, and **pgvector**. It receives ServiceNow tickets via webhooks, analyzes them using AI, retrieves similar past resolutions from a vector database, and posts an AI-generated resolution plan back to ServiceNow.

## Architecture

```
ServiceNow Webhook
        │
        ▼
   ┌─────────┐
   │ FastAPI  │  ← Validates incoming tickets (Pydantic)
   └────┬────┘
        │ background task
        ▼
┌──────────────────────────── LangGraph Workflow ──────────────────────────────┐
│                                                                              │
│  summarize_ticket  →  retrieve_context  →  generate_solution  →  post_to_sn │
│       (Gemini)         (pgvector)             (Gemini)          (REST API)   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Graph Nodes

| Node | Purpose |
|---|---|
| `summarize_ticket` | Extracts the core technical issue from noisy ticket data using Gemini |
| `retrieve_context` | Searches pgvector for similar historically resolved tickets |
| `generate_solution` | Generates a structured resolution plan using the summary + historical context |
| `post_to_servicenow` | Pushes the AI-generated plan back to the ServiceNow ticket as a work note |

### Ingestion Pipeline

Resolved tickets are ingested via separate endpoints, summarized by Gemini, and stored as embeddings in pgvector — building the knowledge base that powers the retrieval step.

## Project Structure

```
now-resolve/
├── main.py                          # FastAPI app + webhook endpoints
├── graph/
│   └── workflow.py                  # LangGraph state graph definition
├── nodes/
│   ├── summarize_ticket_node.py     # Ticket summarization node
│   ├── retrieve_historical_context_node.py  # Vector similarity search node
│   ├── generate_resolution_node.py  # AI resolution generation node
│   └── post_result_to_servicenow_node.py    # ServiceNow API push node
├── states/
│   └── analyzer_state.py            # LangGraph state schema (TypedDict)
├── models/
│   └── ticket_models.py             # Pydantic models for ticket validation
├── utils/
│   ├── db_utils.py                  # Database + embeddings singletons
│   └── insert_ticket_util.py        # Resolved ticket ingestion logic
├── Dockerfile                       # Production container setup
├── requirements.txt                 # Pinned Python dependencies
└── .env                             # Environment variables (not in git)
```

## Prerequisites

- **Python 3.13+**
- **Docker** (for PostgreSQL + pgvector)
- **Google Gemini API key**
- **ServiceNow instance** (for webhook integration)

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/rushikeshnarwade/resolveiq.git
cd resolveiq
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start PostgreSQL with pgvector

```bash
docker run -d \
  --name resolveiq-db \
  -e POSTGRES_USER=admin \
  -e POSTGRES_PASSWORD=admin \
  -e POSTGRES_DB=analyzer \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# Google Gemini
GEMINI_API_KEY=your_gemini_api_key

# PostgreSQL (pgvector)
DATABASE_URL=postgresql://admin:admin@localhost:5432/analyzer

# ServiceNow
SNOW_INSTANCE=your_instance_name
SNOW_USERNAME=your_username
SNOW_PASSWORD=your_password
```

### 4. Start the server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/incident/new` | Receive a new incident → analyze → post resolution |
| `POST` | `/change/new` | Receive a new change request → analyze → post resolution |
| `POST` | `/incident/resolved` | Ingest a resolved incident into the knowledge base |
| `POST` | `/change/resolved` | Ingest a resolved change request into the knowledge base |
| `POST` | `/ingest/batch` | Bulk ingest an array of resolved tickets |

## Deployment

### Docker

```bash
docker build -t resolveiq .
docker run -p 8000:8000 --env-file .env resolveiq
```

### Google Cloud Run

```bash
gcloud run deploy resolveiq \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

Set environment variables via the Cloud Run console or `--set-env-vars`.

## Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** — Async web framework
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — Stateful AI workflow orchestration
- **[Gemini 2.5 Flash](https://ai.google.dev/)** — LLM for summarization and resolution generation
- **[pgvector](https://github.com/pgvector/pgvector)** — Vector similarity search on PostgreSQL
- **[LangChain](https://python.langchain.com/)** — LLM chain composition and embeddings
- **[Pydantic](https://docs.pydantic.dev/)** — Request validation and data modeling