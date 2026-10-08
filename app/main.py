import re
import logging
import uuid
import time
from contextlib import asynccontextmanager
from html import escape
from threading import Lock
from urllib.parse import quote, urlsplit

import gradio as gr
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.rag.chain import build_rag_chain, format_docs_with_sources, index_available
from app.schemas.chat import ChatRequest, ChatResponse, Source
from app.agent.graph import build_agent_graph
from app.agent.guardrails import GuardrailError, check_input, check_output
from app.agent.tools import set_rag_resources
from app.schemas.agent import AgentRequest, AgentResponse, Source as AgentSource, TraceStep


_chain = None
_retriever = None
_rag_lock = Lock()
_agent_graph = None
_agent_checkpointer = None
logger = logging.getLogger(__name__)

# LaTeX delimiters для Gradio Chatbot. LLM-ответы про Ridge, Lasso, метрики
# содержат формулы $$..$$ / \[..\] / $..$ — без этого блока они отрисуются
# как сырые строки `$\ell_1$`.
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chain, _retriever, _agent_graph, _agent_checkpointer
    _ensure_ready()
    _agent_checkpointer = MemorySaver()
    _agent_graph = build_agent_graph(checkpointer=_agent_checkpointer)
    logger.info("Agent graph initialized; RAG ready: %s", _chain is not None)
    try:
        yield
    finally:
        if _retriever is not None:
            _retriever.client.close()
        _chain = None
        _retriever = None
        set_rag_resources(None, None)
        _agent_graph = None
        _agent_checkpointer = None


def _ensure_ready() -> bool:
    global _chain, _retriever
    with _rag_lock:
        try:
            if not index_available():
                _chain = None
                _retriever = None
                set_rag_resources(None, None)
                return False
            if _chain is None or _retriever is None:
                _chain, _retriever = build_rag_chain()
                set_rag_resources(_chain, _retriever)
            return True
        except Exception as exc:
            logger.warning("RAG initialization unavailable (%s)", type(exc).__name__)
            _chain = None
            _retriever = None
            set_rag_resources(None, None)
            return False


def _message_text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        part if isinstance(part, str) else part.get("text", "")
        for part in content
        if isinstance(part, str) or isinstance(part, dict) and part.get("type") == "text"
    )


def _extract_trace(messages: list, sources: list | None = None) -> tuple[list[TraceStep], list[str]]:
    steps: list[TraceStep] = []
    tools_used: list[str] = []
    by_call_id: dict[str, TraceStep] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                step = TraceStep(
                    step=len(steps) + 1,
                    node="agent",
                    tool=tc["name"],
                    input=tc.get("args", {}),
                    output="(tool requested)",
                )
                steps.append(step)
                by_call_id[tc["id"]] = step
        elif isinstance(msg, ToolMessage):
            step = by_call_id.get(msg.tool_call_id)
            if step is not None:
                step.output = _message_text(msg.content)[:500]
                artifact = msg.artifact if isinstance(msg.artifact, dict) else {}
                step.latency_ms = artifact.get("latency_ms")
                if msg.status != "error" and not (
                    step.tool == "documentation_search" and "sources" in artifact
                    and not artifact["sources"]
                ):
                    tools_used.append(step.tool)
    return steps, tools_used


def _extract_sources(messages: list) -> list[AgentSource]:
    sources: list[AgentSource] = []
    search_ids = {tc["id"] for msg in messages if isinstance(msg, AIMessage)
                  for tc in msg.tool_calls if tc["name"] == "documentation_search"}
    for msg in messages:
        if not isinstance(msg, ToolMessage) or msg.status == "error":
            continue
        if msg.name != "documentation_search" and msg.tool_call_id not in search_ids:
            continue
        artifact = msg.artifact if isinstance(msg.artifact, dict) else {}
        if "sources" in artifact:
            sources.extend(AgentSource(**source) for source in artifact["sources"])
            continue
        body, separator, sources_block = _message_text(msg.content).partition("Sources:")
        if separator:
            for line in sources_block.strip().splitlines():
                url = line.strip().removeprefix("- ").strip()
                if urlsplit(url).scheme in ("http", "https", "local"):
                    sources.append(AgentSource(url=url, snippet=body.strip()[:200]))
    return sources


app = FastAPI(title="RAG service", lifespan=lifespan)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/ready")
def ready() -> dict[str, str]:
    if not _ensure_ready():
        raise HTTPException(status_code=503, detail="Active index unavailable")
    return {"status": "ready"}

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not _ensure_ready():
        raise HTTPException(status_code=503, detail="Active index unavailable")
    retriever, chain = _retriever, _chain
    try:
        docs = retriever.invoke(payload.question)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval temporarily unavailable. Raw: {type(exc).__name__}",
        ) from exc
    try:
        answer = chain.invoke({
            "question": payload.question,
            "context": format_docs_with_sources(docs),
        })
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"LLM provider temporarily unavailable. "
                f"Try again in 30-60 seconds. Raw: {type(exc).__name__}"
            ),
        ) from exc
    sources = [
        Source(
            url=doc.metadata.get("source", "unknown"),
            snippet=doc.page_content[:200].strip(),
        )
        for doc in docs
    ]
    return ChatResponse(answer=answer, sources=sources)


@app.post("/agent", response_model=AgentResponse)
def agent_chat(payload: AgentRequest) -> AgentResponse:
    try:
        check_input(payload.question)
    except GuardrailError as e:
        raise HTTPException(status_code=422, detail=f"Input rejected: {e}") from e

    if _agent_graph is None:
        raise HTTPException(status_code=503, detail="Agent graph unavailable")
    thread_id = payload.thread_id or str(uuid.uuid4())
    question_message = HumanMessage(content=payload.question, id=str(uuid.uuid4()))
    t0 = time.perf_counter()
    try:
        result = _agent_graph.invoke(
            {"messages": [question_message], "iteration_count": 0},
            config={"configurable": {"thread_id": thread_id},
                    "recursion_limit": 2 * settings.max_iterations + 2},
        )
    except Exception as exc:
        logger.warning("Agent execution failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail=f"Agent temporarily unavailable. Raw: {type(exc).__name__}",
        ) from exc
    total_ms = int((time.perf_counter() - t0) * 1000)

    messages = result["messages"]
    current_start = next(i for i, msg in enumerate(messages) if msg.id == question_message.id)
    current_messages = messages[current_start + 1:]
    final = current_messages[-1] if current_messages else None
    if not isinstance(final, AIMessage) or final.tool_calls or not _message_text(final.content).strip():
        raise HTTPException(status_code=503, detail="Agent did not produce a final answer")
    raw_answer = _message_text(final.content)
    trace_steps, tools_used = _extract_trace(current_messages)

    safe_answer, guardrail_reason = check_output(raw_answer, tools_used)

    return AgentResponse(
        answer=safe_answer,
        trace=trace_steps,
        sources=_extract_sources(current_messages),
        guardrail_triggered=guardrail_reason,
        iterations=result.get("iteration_count", 0),
        thread_id=thread_id,
        total_ms=total_ms,
        termination_reason=result.get("termination_reason"),
    )


def _format_timings(
    status: str,
    retrieval_ms: float | None = None,
    ttft_ms: float | None = None,
    llm_ms: float | None = None,
    total_ms: float | None = None,
) -> str:
    lines = ["### ⏱ Последний запрос", "", f"**{status}**", ""]
    if retrieval_ms is not None:
        lines.append(f"- 🔍 **Поиск:** {retrieval_ms:.0f} мс")
    if ttft_ms is not None:
        lines.append(f"- ⚡ **Первый фрагмент LLM:** {ttft_ms:.0f} мс")
    if llm_ms is not None and ttft_ms is not None:
        lines.append(f"- 🤖 **Генерация LLM:** {llm_ms - ttft_ms:.0f} мс")
    if total_ms is not None:
        lines.append(f"- 📊 **Всего:** {total_ms:.0f} мс")
    return "\n".join(lines)


def _escape_markdown(text: str) -> str:
    return re.sub(r"([\\`*_\[\]])", r"\\\1", escape(text, quote=False))


def _format_sources(docs: list) -> str:
    if not docs:
        return "### 📚 Источники\n\n_Ничего не найдено_"
    lines = ["### 📚 Источники", ""]
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        title = _escape_markdown(doc.metadata.get("title") or source)
        parsed = urlsplit(source)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            url = quote(source, safe=":/?#[]@!$&'()*+,;=%")
            title = f"[{title}](<{url}>)"
        snippet = _escape_markdown(doc.page_content[:140].strip().replace("\n", " "))
        if len(doc.page_content) > 140:
            snippet += "…"
        lines.append(f"**[{i}]** {title}")
        lines.append(f"> {snippet}")
        lines.append("")
    return "\n".join(lines)


def respond(message: str, history: list):
    """Streaming Gradio handler — generator that yields on every chunk."""
    if not message or not message.strip():
        yield history, "", "### ⏱ Тайминги\n\n_Пустой запрос_", "### 📚 Источники\n\n_—_"
        return

    started = time.perf_counter()
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    empty_sources = "### 📚 Источники\n\n_—_"
    yield history, "", _format_timings("Проверка готовности…"), empty_sources

    if not _ensure_ready():
        history = history[:-1] + [{"role": "assistant", "content": "⚠️ Индекс пока недоступен. Попробуй позже."}]
        yield history, "", _format_timings(
            "Ошибка: индекс недоступен", total_ms=(time.perf_counter() - started) * 1000,
        ), empty_sources
        return

    retriever, chain = _retriever, _chain

    yield history, "", _format_timings("Поиск по документации…"), empty_sources
    t0 = time.perf_counter()
    try:
        docs = retriever.invoke(message)
    except Exception as exc:
        ended = time.perf_counter()
        history = history[:-1] + [{"role": "assistant", "content": f"⚠️ Поиск сейчас недоступен ({type(exc).__name__})."}]
        yield history, "", _format_timings(
            "Ошибка поиска", retrieval_ms=(ended - t0) * 1000,
            total_ms=(ended - started) * 1000,
        ), empty_sources
        return
    retrieval_ms = (time.perf_counter() - t0) * 1000
    sources_panel = _format_sources(docs)

    yield history, "", _format_timings("Ожидание первого фрагмента LLM…", retrieval_ms), sources_panel

    t1 = time.perf_counter()
    ttft_ms: float | None = None
    accumulated = ""
    try:
        for chunk in chain.stream({
            "question": message,
            "context": format_docs_with_sources(docs),
        }):
            if not chunk:
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t1) * 1000
            accumulated += chunk
            history = history[:-1] + [{"role": "assistant", "content": accumulated}]
            yield (
                history, "",
                _format_timings("Генерация ответа…", retrieval_ms, ttft_ms),
                sources_panel,
            )

        ended = time.perf_counter()
        if ttft_ms is None:
            history = history[:-1] + [{"role": "assistant", "content": "⚠️ LLM не вернула ответ. Попробуй ещё раз."}]
            yield history, "", _format_timings(
                "Ошибка: пустой ответ LLM", retrieval_ms, total_ms=(ended - started) * 1000,
            ), sources_panel
            return

        llm_total_ms = (ended - t1) * 1000
        yield (
            history, "",
            _format_timings("Завершено", retrieval_ms, ttft_ms, llm_total_ms, (ended - started) * 1000),
            sources_panel,
        )

    except Exception as exc:
        ended = time.perf_counter()
        error = (
            f"⚠️ LLM-провайдер сейчас недоступен ({type(exc).__name__}). "
            f"Попробуй через 30-60 секунд."
        )
        content = f"{accumulated}\n\nОтвет не завершён. {error}" if accumulated else error
        history = history[:-1] + [{"role": "assistant", "content": content}]
        yield (
            history, "",
            _format_timings("Ошибка LLM", retrieval_ms, ttft_ms, (ended - t1) * 1000, (ended - started) * 1000),
            sources_panel,
        )


def _respond_ui(message: str, history: list):
    """Keep controls disabled throughout streaming, including handled failures."""
    try:
        for chat_history, value, timings, sources in respond(message, history):
            yield chat_history, gr.update(value=value, interactive=False), timings, sources, gr.update(interactive=False)
    except Exception:
        yield gr.skip(), gr.update(interactive=False), _format_timings(
            "Ошибка обработки запроса. Попробуй ещё раз.",
        ), gr.skip(), gr.update(interactive=False)
    yield gr.skip(), gr.update(interactive=True), gr.skip(), gr.skip(), gr.update(interactive=True)


CSS = """
.gradio-container { max-width: 100% !important; padding: 1rem !important; }
#chatbot { height: calc(100vh - 220px) !important; min-height: 500px !important; }
#side-panel { height: calc(100vh - 220px) !important; overflow-y: auto !important;
              padding: 1rem !important; border-left: 1px solid #ddd !important; }
"""

with gr.Blocks(
    title="scikit-learn docs RAG",
    fill_height=True,
) as demo:
    gr.Markdown(
        "# 📖 scikit-learn docs RAG assistant\n"
        "_Спрашивай про модели, оценку, подготовку данных и подбор параметров — на русском или английском._"
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                latex_delimiters=LATEX_DELIMITERS,
                buttons=["copy"],
                avatar_images=(None, None),
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Например: «Покажи формулу Ridge» или «Чем precision отличается от recall»",
                    scale=8,
                    container=False,
                    autofocus=True,
                )
                send = gr.Button("Отправить", scale=1, variant="primary")
            gr.Examples(
                examples=[
                    "How does Ridge regression work?",
                    "Что ты умеешь?",
                    "Объясни разницу между precision и recall с формулами",
                    "When does a decision tree overfit?",
                ],
                inputs=msg,
            )
            with gr.Column(scale=1, elem_id="side-panel"):
                timings_md = gr.Markdown(
                    "### ⏱ Тайминги последнего запроса\n\n_Задайте вопрос, чтобы увидеть тайминги._"
                )
                sources_md = gr.Markdown("### 📚 Источники\n\n_—_")

    gr.on(
        triggers=[msg.submit, send.click],
        fn=_respond_ui,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, timings_md, sources_md, send],
        trigger_mode="once",
        concurrency_limit=1,
        concurrency_id="chat",
        show_progress="minimal",
        api_name="respond",
    )

demo.queue(max_size=8, default_concurrency_limit=1)

app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    theme=gr.themes.Soft(),
    css=CSS,
)
