import pytest
from unittest.mock import MagicMock, patch

from app.agent.tools import documentation_search, python_repl, web_search
from langchain_core.documents import Document


@patch("app.agent.tools.build_rag_chain")
def test_documentation_search_returns_answer_and_sources(mock_build):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Ridge is L2 regularization."
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        MagicMock(metadata={"source": "https://scikit-learn.org/ridge.html"})
    ]
    mock_build.return_value = (mock_chain, mock_retriever)

    import app.agent.tools as tools_module
    tools_module._chain = None
    tools_module._retriever = None

    result = documentation_search.invoke({"query": "what is Ridge?"})

    assert "Ridge" in result
    assert "Sources:" in result
    assert "scikit-learn.org" in result


def test_python_repl_executes_arithmetic():
    result = python_repl.invoke({"code": "print(7 * 6)"})
    assert "42" in result


def test_python_repl_handles_syntax_error_gracefully():
    result = python_repl.invoke({"code": "print(1 +"})
    assert "error" in result.lower() or "syntax" in result.lower()


def test_web_search_respects_disable_flag(monkeypatch):
    monkeypatch.setattr("app.agent.tools.settings.enable_web_search", False)
    result = web_search.invoke({"query": "latest news"})
    assert "disabled" in result.lower()


def test_web_search_returns_titles_and_source_urls(monkeypatch):
    import app.agent.tools as tools_module
    wrapper = MagicMock()
    wrapper.run.return_value = "Snippet without a source."
    wrapper.results.return_value = [{
        "title": "Ridge API", "snippet": "alpha: default=1.0",
        "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html",
    }]
    monkeypatch.setattr(tools_module._ddg_search, "api_wrapper", wrapper)
    monkeypatch.setattr(tools_module.settings, "enable_web_search", True)
    result = web_search.invoke({"query": "Ridge default alpha"})
    assert "Ridge API" in result
    assert wrapper.results.return_value[0]["link"] in result
    wrapper.results.assert_called_once_with("Ridge default alpha", 3, source="text")


def test_documentation_tool_preserves_original_chunks_and_timing(monkeypatch):
    import app.agent.tools as tools_module
    retriever, chain = MagicMock(), MagicMock()
    retriever.invoke.return_value = [Document(
        page_content="Original Ridge passage.", metadata={"source": "https://example.org/ridge"},
    )]
    chain.invoke.return_value = "Generated summary."
    monkeypatch.setattr(tools_module, "_chain", chain)
    monkeypatch.setattr(tools_module, "_retriever", retriever)
    result = documentation_search.invoke({
        "name": "documentation_search", "args": {"query": "Ridge"},
        "id": "doc-artifact", "type": "tool_call",
    })
    assert "Generated summary." in result.content
    assert result.artifact["sources"] == [{
        "url": "https://example.org/ridge", "snippet": "Original Ridge passage.",
        "full_context": "Original Ridge passage.",
    }]
    assert result.artifact["latency_ms"] >= 0


def test_python_tool_records_its_own_timing():
    result = python_repl.invoke({
        "name": "python_repl", "args": {"code": "print(6 * 7)"},
        "id": "python-timing", "type": "tool_call",
    })
    assert result.content.strip() == "42"
    assert result.artifact["latency_ms"] >= 0
