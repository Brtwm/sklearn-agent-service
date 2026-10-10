import re
import json
import logging
import uuid
import time
from contextlib import asynccontextmanager
from html import escape
from threading import Lock
from urllib.parse import quote, urlsplit

import gradio as gr
from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Scope, Receive, Send

from app.config import settings
from app.rag.chain import build_rag_chain, format_docs_with_sources, index_available
from app.schemas.chat import ChatRequest, ChatResponse, Source
from app.agent.graph import build_agent_graph
from app.agent.guardrails import GuardrailError, check_input, check_output
from app.agent.tools import set_rag_resources
from app.schemas.agent import AgentRequest, AgentResponse, Source as AgentSource, TraceStep
from app.core.logging import configure_agent_logging

configure_agent_logging()

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


class DisableProxyBufferingMiddleware:
    """Set queue response headers without collecting or buffering the body."""

    SSE_PATHS = ("/gradio_api/queue/data", "/gradio_api/queue/join", "/queue/data", "/queue/join")

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http" or not scope["path"].startswith(self.SSE_PATHS):
            await self.app(scope, receive, send)
            return

        async def send_unbuffered(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Accel-Buffering"] = "no"
                headers["Cache-Control"] = "no-cache"
            await send(message)

        await self.app(scope, receive, send_unbuffered)


app = FastAPI(title="RAG service", lifespan=lifespan)
app.add_middleware(DisableProxyBufferingMiddleware)

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


def _format_agent_trace(messages: list) -> str:
    steps, _ = _extract_trace(messages)
    if not steps:
        return "_Без вызовов инструментов._"
    lines = []
    for step in steps:
        timing = f"{step.latency_ms} мс" if step.latency_ms is not None else "время не измерено"
        args = _escape_markdown(json.dumps(step.input, ensure_ascii=False))
        output = _escape_markdown(step.output.strip()).replace("\n", "\n> ")
        lines.append(
            f"**{step.step}. {_escape_markdown(step.tool or 'tool')}** · {timing}\n\n"
            f"**Параметры:** {args}\n\n**Результат:**\n\n> {output}"
        )
    return "\n\n---\n\n".join(lines)


def _agent_sources_panel(messages: list) -> str:
    sources = _extract_sources(messages)
    if not sources:
        return "### 📚 Источники\n\n_Источников пока нет._"
    return _format_sources([Document(
        page_content=source.full_context or source.snippet, metadata={"source": source.url},
    ) for source in sources])


def respond_agent(message: str, history: list, thread_id_state: str):
    """Stream the current graph run; the last AIMessage supplies the final answer."""
    empty_sources = "### 📚 Источники\n\n_—_"
    empty_trace = "_Без вызовов инструментов._"
    if not message or not message.strip():
        yield history, "", _format_timings("Пустой запрос"), empty_sources, empty_trace, "💤 Готов к работе", thread_id_state
        return
    try:
        check_input(message)
    except GuardrailError as exc:
        history = history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": f"⚠️ Запрос отклонён: {exc}"}]
        yield history, "", _format_timings("Запрос отклонён"), empty_sources, empty_trace, "🛡 Запрос отклонён", thread_id_state
        return

    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    thread_id = thread_id_state or str(uuid.uuid4())
    started = time.perf_counter()
    yield history, "", _format_timings("Старт агента…"), empty_sources, empty_trace, "🤔 Агент выбирает инструмент…", thread_id

    final_messages = []
    iterations = 0
    termination_reason = None
    accumulated = ""
    generation = None
    try:
        if _agent_graph is None:
            raise RuntimeError("Agent graph unavailable")
        for stream_mode, payload in _agent_graph.stream(
            {"messages": [HumanMessage(content=message, id=str(uuid.uuid4()))], "iteration_count": 0},
            config={"configurable": {"thread_id": thread_id},
                    "recursion_limit": 2 * settings.max_iterations + 2},
            stream_mode=["updates", "messages"],
        ):
            if stream_mode == "messages":
                chunk, metadata = payload
                if metadata.get("langgraph_node") != "agent":
                    continue
                current_generation = (metadata.get("langgraph_step"), chunk.id)
                if current_generation != generation:
                    generation, accumulated = current_generation, ""
                if getattr(chunk, "tool_call_chunks", None):
                    continue
                text = _message_text(chunk.content)
                if not text:
                    continue
                accumulated += text
                history = history[:-1] + [{"role": "assistant", "content": accumulated}]
                status = f"✍️ Агент пишет ответ… {len(accumulated)} симв."
            else:
                node, update = next(iter(payload.items()))
                node_messages = update.get("messages", []) or []
                final_messages = add_messages(final_messages, node_messages)
                if node == "agent":
                    iterations = update.get("iteration_count", iterations + 1)
                    termination_reason = update.get("termination_reason")
                    calls = [call for msg in node_messages if isinstance(msg, AIMessage)
                             for call in msg.tool_calls]
                    if termination_reason:
                        status = "⚠️ Достигнут лимит итераций"
                    elif calls:
                        accumulated = ""
                        history = history[:-1] + [{"role": "assistant", "content": ""}]
                        labels = {"documentation_search": "📚 Поиск документации",
                                  "python_repl": "🧮 Расчёт Python", "web_search": "🌐 Веб-поиск"}
                        status = " · ".join(labels.get(call["name"], call["name"]) for call in calls)
                    else:
                        status = "✍️ Проверяю итоговый ответ…"
                else:
                    names = ", ".join(msg.name or "tool" for msg in node_messages if isinstance(msg, ToolMessage))
                    status = f"🔍 Анализирую результаты: {names}"
            yield (history, "", _format_timings("Агент работает…", total_ms=(time.perf_counter() - started) * 1000),
                   _agent_sources_panel(final_messages), _format_agent_trace(final_messages), status, thread_id)

        final = final_messages[-1] if final_messages else None
        if not isinstance(final, AIMessage) or final.tool_calls or not _message_text(final.content).strip():
            raise RuntimeError("Agent did not produce a final answer")
        steps, tools_used = _extract_trace(final_messages)
        answer, guardrail = check_output(_message_text(final.content), tools_used)
        history = history[:-1] + [{"role": "assistant", "content": answer}]
        total_ms = (time.perf_counter() - started) * 1000
        if termination_reason == "iteration_limit":
            status = "⚠️ Достигнут лимит итераций — задача не завершена"
        elif guardrail:
            status = f"🛡 Ответ изменён проверкой: {guardrail}"
        else:
            status = f"✅ Готово за {total_ms / 1000:.1f} сек · {len(steps)} запросов к инструментам · {iterations} итераций"
        phase = "Задача не завершена" if termination_reason else "Ответ проверен" if guardrail else "Агент завершил работу"
        timings = (_format_timings(phase, total_ms=total_ms)
                   + f"\n- 🔄 **Итераций:** {iterations}\n- 🛠 **Запросов к инструментам:** {len(steps)}")
        yield history, "", timings, _agent_sources_panel(final_messages), _format_agent_trace(final_messages), status, thread_id
    except Exception as exc:
        logger.warning("Agent UI execution failed (%s)", type(exc).__name__)
        _, tools_used = _extract_trace(final_messages)
        partial, _ = check_output(accumulated, tools_used)
        error = f"⚠️ Ответ не завершён: агент сейчас недоступен ({type(exc).__name__}). Попробуй ещё раз."
        history = history[:-1] + [{"role": "assistant", "content": f"{partial}\n\n{error}" if partial else error}]
        yield (history, "", _format_timings("Ошибка агента", total_ms=(time.perf_counter() - started) * 1000),
               _agent_sources_panel(final_messages), _format_agent_trace(final_messages), "⚠️ Ошибка агента — ответ не завершён", thread_id)


def _route_respond(message, history, mode, thread_id_state):
    if mode == "Агент (/agent)":
        yield from respond_agent(message, history, thread_id_state)
    else:
        for out in respond(message, history):
            phase = next((line.strip("*") for line in out[2].splitlines() if line.startswith("**")), "Обработка запроса…")
            icon = "✅" if phase == "Завершено" else "⚠️" if phase.startswith("Ошибка") else "⏳"
            yield (*out, "_Быстрый режим: шаги агента не используются._", f"{icon} {phase}", thread_id_state)


def _clear_ui():
    return ([], "", _format_timings("Новый диалог"), "### 📚 Источники\n\n_—_",
            "_Шаги появятся после запроса к агенту._", "💤 Готов к работе", "")


def _respond_ui(message: str, history: list, mode: str = "Быстрый (/chat)", thread_id_state: str = ""):
    """Keep controls disabled throughout streaming, including handled failures."""
    try:
        for chat_history, value, timings, sources, trace, status, thread_id in _route_respond(message, history, mode, thread_id_state):
            yield (chat_history, gr.update(value=value, interactive=False), timings, sources, trace, status,
                   thread_id, gr.update(interactive=False), gr.update(interactive=False))
    except Exception:
        yield gr.skip(), gr.update(interactive=False), _format_timings(
            "Ошибка обработки запроса. Попробуй ещё раз.",
        ), gr.skip(), gr.skip(), "⚠️ Ошибка обработки запроса", gr.skip(), gr.update(interactive=False), gr.update(interactive=False)
    yield (gr.skip(), gr.update(interactive=True), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
           gr.skip(), gr.update(interactive=True), gr.update(interactive=True))


CSS = """
.gradio-container { max-width: 100% !important; padding: 1rem !important; }
#chatbot { height: calc(100vh - 270px) !important; min-height: 460px !important; }
#status-line { padding: .7rem 1rem; border: 1px solid var(--border-color-primary);
               border-radius: 12px; background: var(--background-fill-secondary); }
#side-panel { max-height: calc(100vh - 190px); overflow-y: auto; flex-wrap: nowrap;
              padding: 1rem; border: 1px solid var(--border-color-primary); border-radius: 12px; }
#side-panel > * { flex-shrink: 0; }
@media (max-width: 900px) {
    #chatbot { height: 55vh !important; min-height: 320px !important; }
    #side-panel { max-height: none; }
}
"""

with gr.Blocks(
    title="scikit-learn docs RAG & agent",
    fill_height=True,
) as demo:
    gr.Markdown(
        "# 📖 scikit-learn · ML assistant\n"
        "_Спрашивай про модели, оценку, подготовку данных и подбор параметров — на русском или английском._"
    )
    thread_id_state = gr.State("")
    status_line = gr.Markdown("💤 Готов к работе", elem_id="status-line")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                show_label=False,
                latex_delimiters=LATEX_DELIMITERS,
                buttons=["copy"],
                avatar_images=(None, None),
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Например: «Покажи формулу Ridge» или «Чем precision отличается от recall»",
                    scale=8,
                    container=False,
                    show_label=False,
                    autofocus=True,
                )
                send = gr.Button("Отправить", scale=1, variant="primary")
            gr.Examples(
                label="Примеры вопросов",
                examples=[
                    "How does Ridge regression work?",
                    "Что ты умеешь?",
                    "Объясни разницу между precision и recall с формулами",
                    "When does a decision tree overfit?",
                ],
                inputs=msg,
            )
        with gr.Column(scale=1, min_width=300, elem_id="side-panel"):
            mode_radio = gr.Radio(
                choices=["Быстрый (/chat)", "Агент (/agent)"], value="Быстрый (/chat)", label="Режим",
            )
            gr.Markdown("**Быстрый** — ответ по документации.\n\n**Агент** — документация, расчёты, веб-поиск и память диалога.")
            timings_md = gr.Markdown(
                "### ⏱ Тайминги последнего запроса\n\n_Задайте вопрос, чтобы увидеть тайминги._"
            )
            sources_md = gr.Markdown("### 📚 Источники\n\n_—_")
            with gr.Accordion("Что сделал агент", open=False):
                trace_md = gr.Markdown("_Включите режим «Агент», чтобы увидеть шаги._")

    gr.on(
        triggers=[msg.submit, send.click],
        fn=_respond_ui,
        inputs=[msg, chatbot, mode_radio, thread_id_state],
        outputs=[chatbot, msg, timings_md, sources_md, trace_md, status_line, thread_id_state, send, mode_radio],
        trigger_mode="once",
        concurrency_limit=1,
        concurrency_id="chat",
        show_progress="minimal",
        api_name="respond",
    )
    chatbot.clear(
        fn=_clear_ui,
        outputs=[chatbot, msg, timings_md, sources_md, trace_md, status_line, thread_id_state],
        concurrency_limit=1, concurrency_id="chat", api_name=False,
    )

demo.queue(max_size=8, default_concurrency_limit=1)

app = gr.mount_gradio_app(
    app,
    demo,
    path="/",
    theme=gr.themes.Soft(),
    css=CSS,
)
