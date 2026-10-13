from unittest.mock import MagicMock, patch

import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import build_agent_graph
from app.config import settings


@pytest.mark.parametrize("question", [
    "What is the difference between Lasso and Ridge?",
    "When is recall more important than precision?",
    "Что такое L2-регуляризация?",
    "Что такое F1-мера?",
])
@patch("app.agent.graph._get_llm_with_tools")
def test_indexed_question_requires_documentation_on_first_iteration(mock_factory, question):
    from app.agent.graph import agent_node

    llm = mock_factory.return_value
    llm.invoke.return_value = AIMessage(content="", tool_calls=[{
        "name": "documentation_search", "args": {"query": question}, "id": "required-doc",
    }])
    agent_node({"messages": [HumanMessage(content=question)], "iteration_count": 0})
    assert llm.invoke.call_args.kwargs["tool_choice"] == {
        "type": "function", "function": {"name": "documentation_search"},
    }


@pytest.mark.parametrize("question", [
    "Compute 7 * 6 with python_repl.", "Что ты умеешь?", "Привет!",
    "Search the web for the latest Ridge changes.",
])
@patch("app.agent.graph._get_llm_with_tools")
def test_non_documentation_requests_keep_automatic_tool_choice(mock_factory, question):
    from app.agent.graph import agent_node

    mock_factory.return_value.invoke.return_value = AIMessage(content="Answer")
    agent_node({"messages": [HumanMessage(content=question)], "iteration_count": 0})
    assert "tool_choice" not in mock_factory.return_value.invoke.call_args.kwargs


@patch("app.agent.graph._get_llm_with_tools")
def test_documentation_choice_is_released_after_first_iteration(mock_factory):
    from app.agent.graph import agent_node

    mock_factory.return_value.invoke.return_value = AIMessage(content="Ridge answer")
    agent_node({"messages": [HumanMessage(content="Explain Ridge"),
                             ToolMessage(content="Ridge evidence", name="documentation_search",
                                         tool_call_id="doc")], "iteration_count": 1})
    assert "tool_choice" not in mock_factory.return_value.invoke.call_args.kwargs


@patch("app.agent.graph._get_llm_with_tools")
def test_provider_cannot_silently_ignore_required_documentation(mock_factory):
    from app.agent.graph import agent_node

    mock_factory.return_value.invoke.return_value = AIMessage(content="An unsupported Ridge answer")
    with pytest.raises(RuntimeError, match="required documentation_search"):
        agent_node({"messages": [HumanMessage(content="Explain Ridge")], "iteration_count": 0})

@patch("app.agent.graph._get_llm_with_tools")
def test_graph_terminates_after_max_iterations(mock_llm_factory):
    """If LLM keeps returning tool_calls, graph must stop at max_iterations.

    NOTE: each AIMessage and tool_call must have a UNIQUE id. LangGraph's
    `add_messages` reducer deduplicates messages by id — если возвращать
    один и тот же объект из мока, second iteration «затирает» первое
    сообщение, last_msg оказывается ToolMessage, и should_continue
    выходит в END на iter=2.
    """
    counter = {"i": 0}

    def make_loop_response(*args, **kwargs):
        counter["i"] += 1
        return AIMessage(
            id=f"ai-{counter['i']}",
            content="",
            tool_calls=[{
                "name": "python_repl",
                "args": {"code": "print(1)"},
                "id": f"tc-{counter['i']}",
            }],
        )

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = make_loop_response
    mock_llm_factory.return_value = mock_llm

    graph = build_agent_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content="loop forever")], "iteration_count": 0},
        config={"configurable": {"thread_id": "test-loop"}, "recursion_limit": 50},
    )

    assert result["iteration_count"] == settings.max_iterations
    assert result.get("termination_reason") == "iteration_limit"
    assert result["messages"][-1].content
    assert not result["messages"][-1].tool_calls
    requested = {tc["id"] for msg in result["messages"] if isinstance(msg, AIMessage)
                 for tc in msg.tool_calls}
    resolved = {msg.tool_call_id for msg in result["messages"] if isinstance(msg, ToolMessage)}
    assert requested == resolved


@patch("app.agent.graph._get_llm_with_tools")
def test_graph_keeps_history_but_resets_termination_reason(mock_llm_factory):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content="", tool_calls=[{
            "name": "python_repl", "args": {"code": "print(1)"}, "id": "last-call",
        }]),
        AIMessage(content="Second answer"),
    ]
    mock_llm_factory.return_value = mock_llm
    graph = build_agent_graph()
    with patch.object(settings, "max_iterations", 1):
        graph.invoke({"messages": [HumanMessage(content="First question")], "iteration_count": 0},
                     config={"configurable": {"thread_id": "continue"}})
    result = graph.invoke({"messages": [HumanMessage(content="Second question")], "iteration_count": 0},
                          config={"configurable": {"thread_id": "continue"}})
    assert result["messages"][-1].content == "Second answer"
    assert result.get("termination_reason") is None
    assert result["iteration_count"] == 1
    assert [m.content for m in result["messages"] if isinstance(m, HumanMessage)] == [
        "First question", "Second question",
    ]


@patch("app.agent.graph._get_llm_with_tools")
def test_graph_stream_preserves_course_node_names_and_tool_results(mock_llm_factory):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content="", tool_calls=[{
            "name": "python_repl", "args": {"code": "print(6 * 7)"}, "id": "stream-call",
        }]),
        AIMessage(content="The answer is 42."),
    ]
    mock_llm_factory.return_value = mock_llm
    events = list(build_agent_graph().stream(
        {"messages": [HumanMessage(content="Compute 6 * 7")], "iteration_count": 0},
        config={"configurable": {"thread_id": "stream"}},
        stream_mode=["updates", "messages"],
    ))
    updates = [payload for mode, payload in events if mode == "updates"]
    assert [next(iter(update)) for update in updates] == ["agent", "tool_executor", "agent"]
    result = updates[1]["tool_executor"]["messages"][0]
    assert result.content.strip() == "42"
    assert result.artifact["latency_ms"] >= 0
    assert updates[-1]["agent"]["messages"][-1].content == "The answer is 42."


@patch("app.agent.graph._get_llm_with_tools")
def test_graph_resolves_failed_tool_without_exposing_exception_details(mock_llm_factory):
    from app.agent.tools import documentation_search
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        AIMessage(content="", tool_calls=[{
            "name": "documentation_search", "args": {"query": "Ridge"}, "id": "failure",
        }]),
        AIMessage(content="Documentation search is unavailable."),
    ]
    mock_llm_factory.return_value = mock_llm
    with patch.object(documentation_search, "func", side_effect=TimeoutError("private diagnostic")):
        result = build_agent_graph().invoke(
            {"messages": [HumanMessage(content="Explain Ridge")], "iteration_count": 0},
            config={"configurable": {"thread_id": "failed-tool"}},
        )
    errors = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(errors) == 1
    assert errors[0].status == "error"
    assert errors[0].tool_call_id == "failure"
    assert "TimeoutError" in errors[0].content
    assert "private diagnostic" not in errors[0].content
    assert result["messages"][-1].content == "Documentation search is unavailable."
