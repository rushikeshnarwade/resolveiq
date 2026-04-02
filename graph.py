import os
from functools import lru_cache
from typing import TypedDict, Annotated, Union, Optional
import operator
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langgraph.graph import StateGraph, START, END


class Priority(str, Enum):
    CRITICAL = "1"
    HIGH = "2"
    MODERATE = "3"
    LOW = "4"
    VERY_LOW = "5"


class ChangeType(str, Enum):
    STANDARD = "standard"
    NORMAL = "normal"
    EMERGENCY = "emergency"


class BaseTicket(BaseModel):
    sys_id: str = Field(..., description="Unique identifier for the ticket", pattern="^[a-zA-Z0-9]{32}$")
    assignment_group: Optional[str] = Field(None, description="Assignment group for the ticket")
    assigned_to: Optional[str] = Field(None, description="Person assigned to the ticket")
    cmdb_ci: Optional[str] = Field(None, description="Configuration item associated with the ticket")
    number: str = Field(..., description="Ticket number")
    priority: Priority
    short_description: str = Field(..., description="Short description of the ticket")
    description: Optional[str] = Field(None, description="Description of the ticket")


class IncidentTicket(BaseTicket):
    category: str = Field(..., min_length=2, description="Category of the incident")
    caller_id: str = Field(..., description="Caller ID for the incident")
    severity: str = Field(..., description="Impact severity")


class ChangeRequestTicket(BaseTicket):
    type: ChangeType
    risk: str = Field(..., description="Risk level of the change")
    justification: str = Field(..., description="Why is this change needed?")
    implementation_plan: str = Field(..., description="How will the change be implemented?")


class AnalyzerState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    ticket: Union[IncidentTicket, ChangeRequestTicket]
    clean_summary: str
    past_solutions: list[str]
    proposed_plan: str


# ----------------------------
# LLM NODES
# ----------------------------

@lru_cache(maxsize=1)
def _get_summarization_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Extract the core technical issue from the ServiceNow ticket data. "
            "Ignore noise. Output a dense brief technical summary."
        ),
        ("human", "Short Description: {short_desc}\n\nFull Description: {full_desc}")
    ])
    return prompt | llm | StrOutputParser()


def summarize_ticket_node(state: AnalyzerState) -> dict:
    ticket = state["ticket"]
    full_desc = ticket.description or ""
    clean_summary = _get_summarization_chain().invoke({
        "short_desc": ticket.short_description,
        "full_desc": full_desc
    })
    return {"clean_summary": clean_summary}


@lru_cache(maxsize=1)
def _get_solution_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strictly professional Senior IT Solutions Architect.

CRITICAL GUARDRAILS:
1. You must ONLY answer IT, networking, software, or hardware-related queries.
2. If the user's ticket is a joke, spam, or non-IT related, reply EXACTLY with:
   "REJECTED: Non-IT related query detected."
3. Do NOT make up hypothetical server names or IP addresses. Only use what is provided in the context.

FORMATTING RULES:
You must format your response exactly like this:

### 🎯 Issue Analysis
[1-2 sentences explaining the root cause based on the summary]

### 🛠️ Proposed Resolution Steps
1. [Step 1]
2. [Step 2]
3. [Step 3]

### 📚 Historical Context
[Briefly mention if past tickets helped form this solution, or state if no relevant history was found.]
"""),
        ("human", "Current Issue Summary: {summary}\n\nPast Successful Solutions:\n{past_solutions}")
    ])
    return prompt | llm | StrOutputParser()


def generate_solution_node(state: AnalyzerState) -> dict:
    formatted_past_solutions = "\n".join([f"- {sol}" for sol in state["past_solutions"]])
    if not formatted_past_solutions:
        formatted_past_solutions = "No relevant past solutions found in the database."

    final_plan = _get_solution_chain().invoke({
        "summary": state["clean_summary"],
        "past_solutions": formatted_past_solutions
    })

    return {"proposed_plan": final_plan}


# ----------------------------
# LAZY DB / VECTOR HELPERS
# ----------------------------

def _get_pgvector_connection_string() -> str:
    raw_connection_string = os.getenv("DATABASE_URL")
    if not raw_connection_string:
        raise RuntimeError("DATABASE_URL is missing")

    if raw_connection_string.startswith("postgresql://"):
        return raw_connection_string.replace("postgresql://", "postgresql+psycopg://", 1)

    return raw_connection_string


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=768,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name="historical_tickets",
        connection=_get_pgvector_connection_string(),
        use_jsonb=True,
    )


def retrieve_historical_context(state: AnalyzerState) -> dict:
    ticket = state["ticket"]
    search_query = (state["clean_summary"] or ticket.short_description).strip()

    filter_dict = {}
    if ticket.cmdb_ci:
        filter_dict["cmdb_ci"] = ticket.cmdb_ci

    embeddings = get_embeddings()
    vector_store = get_vector_store()

    # Use explicit embedding + vector search to avoid internal batching issues
    query_embedding = embeddings.embed_query(search_query)

    result = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=3,
        filter=filter_dict if filter_dict else None,
    )

    past_solutions = [doc.page_content for doc in result]
    return {"past_solutions": past_solutions}


# ----------------------------
# GRAPH
# ----------------------------

workflow = StateGraph(AnalyzerState)

workflow.add_node("summarize_ticket", summarize_ticket_node)
workflow.add_node("retrieve_context", retrieve_historical_context)
workflow.add_node("generate_solution", generate_solution_node)

workflow.add_edge(START, "summarize_ticket")
workflow.add_edge("summarize_ticket", "retrieve_context")
workflow.add_edge("retrieve_context", "generate_solution")
workflow.add_edge("generate_solution", END)

app_graph = workflow.compile()