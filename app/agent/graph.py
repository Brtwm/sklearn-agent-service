from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
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


_llm_with_tools = None


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = get_llm().bind_tools(TOOLS)
    return _llm_with_tools


def agent_node(state: AgentState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = _get_llm_with_tools().invoke(messages)
    return {
        "messages": [response],
        "iteration_count": state["iteration_count"] + 1,
    }


tool_executor_node = ToolNode(TOOLS)


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