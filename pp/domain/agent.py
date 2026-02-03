from enum import Enum
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentEventType(str, Enum):
    # lifecycle
    Start = "start"
    Done = "done"
    Error = "error"
    
    # streaming
    TextDelta = "text_delta"
    TextComplete = "text_complete"


@dataclass
class AgentEvent:
    type: AgentEventType
    input_data: dict[str, Any] = field(default_factory=dict)
