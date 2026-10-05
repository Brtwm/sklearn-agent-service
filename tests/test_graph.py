from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_agent_graph

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

    assert result["iteration_count"] >= 5