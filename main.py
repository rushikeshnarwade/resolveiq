from dotenv import load_dotenv
load_dotenv()

import os
import requests
from typing import List
from fastapi import FastAPI, Request, BackgroundTasks
import logging
from graph import IncidentTicket, ChangeRequestTicket, app_graph
from models.resolved_ticket_model import ResolvedTicketPayload
from batch_ingest import process_batch

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ServiceNow LangGraph Analyzer")

# UTILITY FUNCTION TO PUSH PROPOSED PLAN BACK TO SERVICENOW
def push_to_servicenow(table_name: str, sys_id: str, proposed_plan: str):
    """
    Acts as an API client to push the AI's plan directly into the 
    work_notes of the active ServiceNow ticket.
    """
    # 1. Grab credentials securely from your .env file
    instance = os.getenv("SNOW_INSTANCE")
    user = os.getenv("SNOW_USERNAME")
    pwd = os.getenv("SNOW_PASSWORD")
    
    if not instance or instance == "dev12345":
        logging.warning("⚠️ Skipping ServiceNow push: Real credentials not configured in .env")
        return

    # 2. Build the exact ServiceNow API endpoint
    # Example: https://dev12345.service-now.com/api/now/table/incident/ab12345...
    url = f"https://{instance}.service-now.com/api/now/table/{table_name}/{sys_id}"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # 3. Format the payload so it looks clean in the ServiceNow UI
    payload = {
        "work_notes": f"🤖 **AI Proposed Resolution Plan** 🤖\n\n{proposed_plan}"
    }

    # 4. Fire the request! We use PATCH to only update the work_notes field.
    try:
        response = requests.patch(url, auth=(user, pwd), headers=headers, json=payload)
        
        if response.status_code in [200, 201]:
            logging.info(f"✅ Successfully pushed work note to {table_name} {sys_id}")
        else:
            logging.error(f"❌ Failed to push to ServiceNow: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"❌ Connection error to ServiceNow: {e}")

# GRAPH EXECUTION WRAPPER
def run_analyzer_graph(ticket_data):
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
    final_state = app_graph.invoke(initial_state, config=config)
    logging.info(f"✅ Analysis Complete for {ticket_data.number}!")
    
    table_name = "incident" if hasattr(ticket_data, 'caller_id') else "change_request"
    
    push_to_servicenow(
        table_name=table_name,
        sys_id=ticket_data.sys_id,
        proposed_plan=final_state['proposed_plan']
    )

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
    validated_ticket = ResolvedTicketPayload(**payload)
    
    logging.info(f"🧠 REAL-TIME LEARNING: Ingesting newly resolved incident {validated_ticket.number}")
    
    # We reuse our batch processor, but just give it a list containing this ONE ticket!
    background_tasks.add_task(process_batch, [validated_ticket])
    
    return {"status": "success", "message": f"Learning from incident {validated_ticket.number}"}, 202

@app.post("/change/resolved")
async def process_resolved_change(
    request: Request, 
    background_tasks: BackgroundTasks
):
    """Accepts a newly resolved change request and adds it to the AI's brain instantly."""
    payload = await request.json()
    
    validated_ticket = ResolvedTicketPayload(**payload)
    
    logging.info(f"🧠 REAL-TIME LEARNING: Ingesting newly resolved change {validated_ticket.number}")
    
    background_tasks.add_task(process_batch, [validated_ticket])
    
    return {"status": "success", "message": f"Learning from change {validated_ticket.number}"}, 202

@app.post("/ingest/batch")
async def process_batch_ingestion(
    tickets: List[ResolvedTicketPayload], 
    background_tasks: BackgroundTasks
):
    """
    Accepts a massive array of resolved tickets from ServiceNow 
    and processes them in the background.
    """
    logging.info(f"📥 Received {len(tickets)} tickets for batch ingestion.")
    
    # We pass the validated list of objects to our background task
    background_tasks.add_task(process_batch, tickets)
    
    # Instantly return 202 so ServiceNow doesn't timeout
    return {"status": "success", "message": f"Batch of {len(tickets)} tickets accepted for processing"}, 202