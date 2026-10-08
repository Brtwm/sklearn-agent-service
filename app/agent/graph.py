from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOLS
from app.config import settings
from app.llm import get_llm


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
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = _get_llm_with_tools().invoke(messages)
    iterations = state["iteration_count"] + 1
    updates = [response]
    termination_reason = None
    if iterations >= settings.max_iterations and response.tool_calls:
        # Resolve every pending call so the checkpoint can continue next turn.
        updates.extend(ToolMessage(
            content="Not executed: agent iteration limit reached.",
            tool_call_id=call["id"], name=call["name"], status="error",
        ) for call in response.tool_calls)
        question = next((m.content for m in reversed(state["messages"])
                         if isinstance(m, HumanMessage)), "")
        russian = isinstance(question, str) and any("А" <= c <= "я" or c in "Ёё" for c in question)
        updates.append(AIMessage(content=(
            "Достигнут лимит итераций агента. Задача не завершена; попробуйте разбить вопрос на части."
            if russian else
            "Agent iteration limit reached. The task is incomplete; try splitting the question into smaller parts."
        )))
        termination_reason = "iteration_limit"
    return {
        "messages": updates,
        "iteration_count": iterations,
        "termination_reason": termination_reason,
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
