import os
import logging
import requests

from states import AnalyzerState


def _push_to_servicenow(table_name: str, sys_id: str, proposed_plan: str):
    """
    Acts as an API client to push the AI's plan directly into the 
    work_notes of the active ServiceNow ticket.
    """

    instance = os.getenv("SNOW_INSTANCE")
    user = os.getenv("SNOW_USERNAME")
    pwd = os.getenv("SNOW_PASSWORD")
    
    if not instance or instance == "dev12345":
        logging.warning("⚠️ Skipping ServiceNow push: Real credentials not configured in .env")
        return

    url = f"https://{instance}.service-now.com/api/now/table/{table_name}/{sys_id}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "work_notes": f"🤖 **AI Proposed Resolution Plan** 🤖\n\n{proposed_plan}"
    }

    try:
        response = requests.patch(url, auth=(user, pwd), headers=headers, json=payload)
        
        if response.status_code in [200, 201]:
            logging.info(f"✅ Successfully pushed work note to {table_name} {sys_id}")
        else:
            logging.error(f"❌ Failed to push to ServiceNow: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"❌ Connection error to ServiceNow: {e}")


def post_result_to_servicenow_node(state: AnalyzerState) -> dict:
    """
    LangGraph node: Reads the proposed plan from state and pushes it
    back to the originating ServiceNow ticket as a work note.
    """
    ticket = state["ticket"]
    proposed_plan = state["proposed_plan"]

    # Determine which ServiceNow table this ticket belongs to
    table_name = "incident" if hasattr(ticket, "caller_id") else "change_request"

    _push_to_servicenow(
        table_name=table_name,
        sys_id=ticket.sys_id,
        proposed_plan=proposed_plan,
    )

    return {}
