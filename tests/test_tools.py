import pytest
from unittest.mock import MagicMock, patch

from app.agent.tools import documentation_search, python_repl, web_search


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