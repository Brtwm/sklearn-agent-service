import time
from threading import Lock

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_experimental.tools.python.tool import PythonREPLTool

from app.rag.chain import build_rag_chain, format_docs_with_sources
from app.config import settings

_chain = None
_retriever = None
_python_repl = PythonREPLTool()
_ddg_search = DuckDuckGoSearchResults(num_results=3, output_format="json")
_rag_lock = Lock()


def set_rag_resources(chain, retriever) -> None:
    """Share the service's initialized RAG instead of loading a second embedder."""
    global _chain, _retriever
    with _rag_lock:
        _chain, _retriever = chain, retriever


def _ensure_rag_ready():
    global _chain, _retriever
    with _rag_lock:
        if _chain is None or _retriever is None:
            _chain, _retriever = build_rag_chain()


@tool(response_format="content_and_artifact")
def documentation_search(query: str) -> tuple[str, dict]:
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
    started = time.perf_counter()
    _ensure_rag_ready()
    retriever, chain = _retriever, _chain
    docs = retriever.invoke(query)
    answer = chain.invoke({
        "question": query,
        "context": format_docs_with_sources(docs)
    })
    sources = [doc.metadata.get("source", "unknown") for doc in docs]
    sources_block = "\n".join(f"- {url}" for url in sources)
    return f"{answer}\n\nSources:\n{sources_block}", {
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "sources": [{
            "url": doc.metadata.get("source", "unknown"),
            "snippet": doc.page_content[:200].strip(),
            "full_context": doc.page_content,
        } for doc in docs],
    }


@tool(response_format="content_and_artifact")
def python_repl(code: str) -> tuple[str, dict]:
    """Execute Python code and return the printed output.

    Use this tool for arithmetic, computing formulas, or transforming data.
    The REPL runs in the service process, without a security sandbox.
    Use `print()` to surface results.

    Args:
        code: Python source code to execute.

    Returns:
        Stdout of the executed code, or error message.
    """
    started = time.perf_counter()
    output = _python_repl.run(code)
    return output, {"latency_ms": int((time.perf_counter() - started) * 1000)}


# SECURITY NOTE: PythonREPLTool executes arbitrary code in the same process
# as our app. Agent-mediated invocation does not provide isolation.
# For production with untrusted users, wrap
# in a sandbox (gvisor, firecracker, e2b.dev).
@tool(response_format="content_and_artifact")
def web_search(query: str) -> tuple[str, dict]:
    """Search the web via DuckDuckGo for recent or general-knowledge info.

    Use this tool when the question requires fresh data (release notes,
    latest versions, news) that is NOT in the documentation corpus.

    Args:
        query: free-text search query.

    Returns:
        Up to three search results with titles, snippets and source URLs.
    """
    started = time.perf_counter()
    output = (_ddg_search.run(query) if settings.enable_web_search
              else "Web search is disabled in this environment")
    return output, {"latency_ms": int((time.perf_counter() - started) * 1000)}


TOOLS = [documentation_search, python_repl, web_search]
