from dotenv import load_dotenv
load_dotenv()

from typing import List, Union
from fastapi import FastAPI, Request, BackgroundTasks
import logging
from models.ticket_models import IncidentTicket, ChangeRequestTicket, ResolvedTicket
from graph.workflow import workflow


logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ServiceNow LangGraph Analyzer")

# GRAPH EXECUTION WRAPPER
def run_analyzer_graph(ticket_data: Union[IncidentTicket, ChangeRequestTicket]):
    """This function runs in the background so we don't block ServiceNow"""
    logging.info(f"🚀 Starting LangGraph analysis for {ticket_data.number}...")
    
    initial_state = {
        "messages": [],
        "ticket": ticket_data,
        "clean_summary": "",
        "past_solutions": [],
        "proposed_plan": ""
    }
    
    # Run the graph WITH the thread_id config
    config = {"configurable": {"thread_id": ticket_data.sys_id}}
    workflow.invoke(initial_state, config=config)
    logging.info(f"✅ Analysis Complete for {ticket_data.number}!")

# WEBHOOK ENDPOINTS
@app.post("/incident/new")
async def process_new_incident(request: Request, background_tasks: BackgroundTasks):
    """This endpoint accepts a new incident from ServiceNow"""

    payload = await request.json()
    logging.info(f"🚨 NEW INCIDENT RECEIVED:\n{payload}")
    validated_incident = IncidentTicket(**payload)
    logging.info(f"✅ INCIDENT VALIDATED: {validated_incident.number}")

    background_tasks.add_task(run_analyzer_graph, validated_incident)

    return {"status": "success", "message": "New incident payload received"}, 202

@app.post("/change/new")
async def process_new_change(request: Request, background_tasks: BackgroundTasks):
    """This endpoint is accepts a new change request from ServiceNow"""

    payload = await request.json()
    logging.info(f"🔄 NEW CHANGE REQUEST RECEIVED:\n{payload}")
    validated_change = ChangeRequestTicket(**payload)
    logging.info(f"✅ CHANGE REQUEST VALIDATED: {validated_change.number}")
    background_tasks.add_task(run_analyzer_graph, validated_change)

    return {"status": "success", "message": "New change request payload received"}, 202

@app.post("/incident/resolved")
async def process_resolved_incident(
    request: Request, 
    background_tasks: BackgroundTasks
):
    """Accepts a newly resolved incident and adds it to the AI's brain instantly."""
    payload = await request.json()
    
    # Validate the incoming data using the schema we built in Step 1
    validated_ticket = ResolvedTicket(**payload)
    
    logging.info(f"🧠 REAL-TIME LEARNING: Ingesting newly resolved incident {validated_ticket.number}")
    
    # We reuse our batch processor, but just give it a list containing this ONE ticket!
    background_tasks.add_task(process_tickets, [validated_ticket])
    
    return {"status": "success", "message": f"Learning from incident {validated_ticket.number}"}, 202

@app.post("/change/resolved")
async def process_resolved_change(
    request: Request, 
    background_tasks: BackgroundTasks
):
    """Accepts a newly resolved change request and adds it to the AI's brain instantly."""
    payload = await request.json()
    
    validated_ticket = ResolvedTicket(**payload)
    
    logging.info(f"🧠 REAL-TIME LEARNING: Ingesting newly resolved change {validated_ticket.number}")
    
    background_tasks.add_task(process_tickets, [validated_ticket])
    
    return {"status": "success", "message": f"Learning from change {validated_ticket.number}"}, 202

@app.post("/ingest/batch")
async def process_batch_ingestion(
    tickets: List[ResolvedTicket], 
    background_tasks: BackgroundTasks
):
    """
    Accepts a massive array of resolved tickets from ServiceNow 
    and processes them in the background.
    """
    logging.info(f"📥 Received {len(tickets)} tickets for batch ingestion.")
    
    # We pass the validated list of objects to our background task
    background_tasks.add_task(process_tickets, tickets)
    
    # Instantly return 202 so ServiceNow doesn't timeout
    return {"status": "success", "message": f"Batch of {len(tickets)} tickets accepted for processing"}, 202