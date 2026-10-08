from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    thread_id: str | None = None


class TraceStep(BaseModel):
    step: int
    node: str
    tool: str | None = None
    input: dict | None = None
    output: str
    latency_ms: int | None = Field(default=None, ge=0)


class Source(BaseModel):
    url: str
    snippet: str
    full_context: str = ""


class AgentResponse(BaseModel):
    answer: str
    trace: list[TraceStep]
    sources: list[Source]
    guardrail_triggered: str | None = None
    iterations: int = 0
    thread_id: str | None = None
    total_ms: int = Field(default=0, ge=0)
    termination_reason: str | None = None
