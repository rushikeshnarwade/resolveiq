from pydantic import BaseModel, Field
from typing import Optional

class ResolvedTicketPayload(BaseModel):
    sys_id: str = Field(..., pattern="^[a-zA-Z0-9]{32}$")
    number: str
    short_description: str
    description: str
    category: str
    cmdb_ci: Optional[str] = None
    close_notes: str