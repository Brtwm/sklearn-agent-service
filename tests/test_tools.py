from unittest.mock import MagicMock, patch

from app.agent.tools import documentation_search

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