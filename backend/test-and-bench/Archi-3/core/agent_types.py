from dataclasses import dataclass, field
from typing import Dict, Any, List
import time

@dataclass
class ToolCall:
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any
    timestamp: float = field(default_factory=time.time)

@dataclass
class AgentAction:
    action_type: str
    action_input: Dict[str, Any]
    reasoning: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
