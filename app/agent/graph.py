from typing import Annotated, NotRequired, TypedDict
import time
import structlog

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOLS
from app.config import settings
from app.llm import get_llm

log = structlog.get_logger(__name__)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    termination_reason: NotRequired[str | None]


_llm_with_tools = None


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = get_llm().bind_tools(TOOLS, temperature=settings.agent_temperature)
    return _llm_with_tools


def agent_node(state: AgentState) -> dict:
    iteration = state["iteration_count"] + 1
    started = time.perf_counter()

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    try:
        response = _get_llm_with_tools().invoke(messages)
    except Exception as exc:
        log.error(
            "agent_node_failed",
            iteration=iteration,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error_type=type(exc).__name__,
        )
        raise

    latency_ms = int((time.perf_counter() - started) * 1000)
    tool_calls = response.tool_calls or []
    updates = [response]
    termination_reason = None
    if iteration >= settings.max_iterations and tool_calls:
        # Resolve pending calls so the checkpoint can continue next turn.
        updates.extend(
            ToolMessage(
                content="Not executed: agent iteration limit reached",
                tool_call_id=call["id"],
                name=call["name"],
                status="error",
            )
            for call in tool_calls
        )

        question = next(
            (
                message.content
                for message in reversed(state["messages"])
                if isinstance(message, HumanMessage)
            ),
            "",
        )
        russian = isinstance(question, str) and any(
            "А" <= character <= "я" or character in "Ёё"
            for character in question
        )

        updates.append(
            AIMessage(
                content=(
                    "Достигнут лимит итераций агента. Задача не завершена; "
                    "попробуйте разбить вопрос на части."
                    if russian
                    else
                    "Agent iteration limit reached. The task is incomplete; "
                    "try splitting the question into smaller parts."
                )
            )
        )
        termination_reason = "iteration_limit"

    log.info(
        "agent_node_completed",
        iteration=iteration,
        latency_ms=latency_ms,
        tool_calls_count=len(tool_calls),
        tools_requested=[call["name"] for call in tool_calls],
        termination_reason=termination_reason,
    )

    return {
        "messages": updates,
        "iteration_count": iteration,
        "termination_reason": termination_reason
    }


def _tool_error_message(exc: Exception) -> str:
    return f"Tool failed ({type(exc).__name__}). No result is available."


tool_executor_node = ToolNode(TOOLS, handle_tool_errors=_tool_error_message)


def should_continue(state: AgentState) -> str:
    if state["iteration_count"] >= settings.max_iterations:
        return END
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tool_executor"
    return END


def build_agent_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tool_executor", "agent")
    return graph.compile(checkpointer=checkpointer)
