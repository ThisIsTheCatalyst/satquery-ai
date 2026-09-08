"""
Execution trace — builds the auditable summary the problem statement requires:
"an auditable execution summary containing the selected task, model/tool
names, and key parameters." This is graded, so keep it structured and
complete rather than free-text.
Owner: Person 4 (Agentic Orchestration + GUI).
"""

import time
from typing import Optional


class ExecutionTrace:
    """Accumulates one structured record per query, then exposes it as a
    plain dict (JSON-serializable) for the API response / UI / report.
    """

    def __init__(self, query: str, mode: str):
        self.query = query
        self.mode = mode
        self.start_time = time.time()
        self.task: Optional[str] = None
        self.tools_used: list = []          # list of {"name":.., "task":.., "params":..}
        self.input_metadata: dict = {}
        self.confidence: Optional[float] = None
        self.warnings: list = []
        self.end_time: Optional[float] = None

    def set_task(self, task: str):
        self.task = task

    def set_input_metadata(self, metadata: dict):
        self.input_metadata = metadata

    def add_tool_call(self, name: str, task: str, params: dict = None):
        self.tools_used.append({
            "tool_name": name,
            "task": task,
            "params": params or {},
        })

    def set_confidence(self, confidence: float):
        self.confidence = confidence

    def add_warning(self, message: str):
        self.warnings.append(message)

    def finalize(self) -> dict:
        self.end_time = time.time()
        return {
            "query": self.query,
            "input_mode": self.mode,
            "input_metadata": self.input_metadata,
            "selected_task": self.task,
            "tools_used": self.tools_used,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "latency_seconds": round(self.end_time - self.start_time, 3),
        }
