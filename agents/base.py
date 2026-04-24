from __future__ import annotations
from typing import Any, Optional

# Intra-package imports (forward refs resolved at runtime)
from ..core.memory import MemorySystem
from ..core.validation import OutputValidator
from ..core.events import EventBus, Event, EventType
from .types import TaskState, AgentState


class BaseAgent:
    """Base class for all specialized sub-agents."""
    
    # 全局事件总线（类级别，所有Agent共享）
    bus = EventBus()
    
    def __init__(self, llm: 'LLMClient', memory: MemorySystem, validator: OutputValidator):
        self.llm = llm
        self.memory = memory
        self.validator = validator
        self.name = self.__class__.__name__
    
    def run(self, task_state: TaskState, expert: 'ExpertConfig', messages: list[dict]) -> str:
        raise NotImplementedError


# Re-export for convenience
__all__ = ['BaseAgent', 'TaskState', 'AgentState']


