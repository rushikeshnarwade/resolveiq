import os # Make sure this is at the top of your file!
from dotenv import load_dotenv

from typing import TypedDict, Annotated, Union, Optional
import operator
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from enum import Enum

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

load_dotenv()

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
    description: str = Field(None, description="Description of the ticket")

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


#### GRAPH NODES ####
# Node 1: The Fast Summarizer
summarizer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the core technical issue from the ServiceNow ticket data. Ignore noise. Output a dense brief technical summary"),
    ("human", "Short Description: {short_desc}\n\nFull Description: {full_desc}")
])
summarization_chain = summary_prompt | summarizer_llm | StrOutputParser()

def summarize_ticket_node(state: AnalyzerState) -> dict:
    """Uses Gemini Flash to clean the incoming ticket."""
    ticket = state["ticket"]
    clean_summary = summarization_chain.invoke({
        "short_desc": ticket.short_description,
        "full_desc": ticket.description
    })

    return {"clean_summary": clean_summary}

# Node 2: The Database Retriever
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=768
)
connection_string = os.getenv("PGVECTOR_URL")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="historical_tickets",
    connection=connection_string,
    use_jsonb=True
)

def retrieve_historical_context(state: AnalyzerState) -> dict:
    """Searched the DB using clean_summary and metadata."""
    ticket = state["ticket"]
    search_query = state["clean_summary"]
    search_query = search_query.strip()

    filter_dict = {}
    if ticket.cmdb_ci:
        filter_dict["cmdb_ci"] = ticket.cmdb_ci
    
    result = vector_store.similarity_search(
        query=search_query,
        k=3,
        filter=filter_dict
    )

    past_solutions = [doc.page_content for doc in result]
    return {"past_solutions": past_solutions}

from langgraph.graph import StateGraph, START, END

# 4. FINAL NODE: The Solution Generator

solution_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

solution_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior IT Solutions Architect.
    Analyze the incoming ticket summary and review how similar issues were solved in the past.
    Provide a clear, step-by-step resolution plan to fix the current issue.
    If the past solutions are irrelevant, state that and provide general troubleshooring steps."""),
    ("human", "Current Issue Summary:{summary}\n\nPast Successful Solutions:\n{past_solutions}")
])

solution_chain = solution_prompt | solution_llm| StrOutputParser()

def generate_solution_node(state: AnalyzerState) -> dict:
    """Generates a proposed plan using Gemini Pro."""

    formatted_past_solutions = "\n".join([f"- {sol}" for sol in state["past_solutions"]])
    if not formatted_past_solutions:
        formatted_past_solutions = "No relevant past solutions found in the database."
    
    final_plan = solution_chain.invoke({
        "summary": state["clean_summary"],
        "past_solutions": formatted_past_solutions
    })

    return {"proposed_plan": final_plan}


workflow = StateGraph(AnalyzerState)

workflow.add_node("summarize_ticket", summarize_ticket_node)
workflow.add_node("retrieve_context", retrieve_historical_context)
workflow.add_node("generate_solution", generate_solution_node)

workflow.add_edge(START, "summarize_ticket")
workflow.add_edge("summarize_ticket", "retrieve_context")
workflow.add_edge("retrieve_context", "generate_solution")
workflow.add_edge("generate_solution", END)

# 1. Define the connection pool for the checkpointer
pool_url = os.getenv("DATABASE_URL")
pool = ConnectionPool(
    conninfo=pool_url,
    max_size=20,
    kwargs={"autocommit": True}
)

# 2. Initialize the PostgresSaver
checkpointer = PostgresSaver(pool)

# 3. Compile the graph, attaching the checkpointer and setting the interrupt
# We tell it to pause BEFORE running the 'generate_solution' node.
checkpointer.setup() # This creates the necessary state tables in your DB automatically!

app_graph = workflow.compile(
    checkpointer=checkpointer,
    #interrupt_before=["generate_solution"] 
)