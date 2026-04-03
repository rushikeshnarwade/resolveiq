from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

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

class ResolvedTicket(BaseTicket):
    category: str
    close_notes: str

class IncidentTicket(BaseTicket):
    category: str = Field(..., min_length=2, description="Category of the incident")
    caller_id: str = Field(..., description="Caller ID for the incident")
    severity: str = Field(..., description="Impact severity")

class ChangeRequestTicket(BaseTicket):
    type: ChangeType
    risk: str = Field(..., description="Risk level of the change")
    justification: str = Field(..., description="Why is this change needed?")
    implementation_plan: str = Field(..., description="How will the change be implemented?")