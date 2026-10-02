from langchain_core.tools import tool

from app.rag.chain import build_rag_chain, format_docs_with_sources

_chain = None
_retriever = None


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