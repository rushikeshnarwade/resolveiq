from typing import Annotated, TypedDict, Union
import operator

from langchain_core.messages import AnyMessage
from models.ticket_models import IncidentTicket, ChangeRequestTicket

class AnalyzerState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    ticket: Union[IncidentTicket, ChangeRequestTicket]
    clean_summary: str
    past_solutions: list[str]
    proposed_plan: str