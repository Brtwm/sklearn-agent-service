from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonREPLTool

from app.rag.chain import build_rag_chain, format_docs_with_sources
from app.config import settings

_chain = None
_retriever = None
_python_repl = PythonREPLTool()
_ddg_search = DuckDuckGoSearchRun()


def _ensure_rag_ready():
    global _chain, _retriever
    if _chain is None:
        _chain, _retriever = build_rag_chain()
    

@tool
def documentation_search(query: str) -> str:
    """Search the indexed scikit-learn documentation and return an answer with sources.

    Use for questions about linear models, decision trees, ensembles,
    model evaluation and metrics, cross-validation, preprocessing,
    pipelines, hyperparameter tuning, missing values, and feature selection.
    Also use for questions about this assistant's documented capabilities.

    The corpus covers selected documentation pages, not all of scikit-learn.
    Information outside the indexed corpus may be unavailable.

    Args:
        query: Natural-language question or keywords, in Russian or English.

    Returns:
        An answer grounded in retrieved documentation, followed by source URLs.
    """
    _ensure_rag_ready()
    docs = _retriever.invoke(query)
    answer = _chain.invoke({
        "question": query,
        "context": format_docs_with_sources(docs)
    })
    sources = [doc.metadata.get("source", "unknown") for doc in docs]
    sources_block = "\n".join(f"- {url}" for url in sources)
    return f"{answer}\n\nSources:\n{sources_block}"


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the printed output.

    Use this tool for arithmetic, computing formulas, or transforming data.
    The code runs in a sandboxed REPL; use `print()` to surface results.

    Args:
        code: Python source code to execute.

    Returns:
        Stdout of the executed code, or error message.
    """
    return _python_repl.run(code)


# SECURITY NOTE: PythonREPLTool executes arbitrary code in the same process
# as our app. It is NEVER exposed directly to end users — only the agent
# decides what to send into it. For production with untrusted users, wrap
# in a sandbox (gvisor, firecracker, e2b.dev).
@tool
def web_search(query: str) -> str:
    """Search the web via DuckDuckGo for recent or general-knowledge info.

    Use this tool when the question requires fresh data (release notes,
    latest versions, news) that is NOT in the documentation corpus.

    Args:
        query: free-text search query.

    Returns:
        Top-3 search results as title + snippet text.
    """
    if not settings.enable_web_search:
        return "Web search is disabled in this environment"
    return _ddg_search.run(query)


TOOLS = [documentation_search, python_repl, web_search]