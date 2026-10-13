"""Run the agent from the command line without starting FastAPI."""

from langchain_core import messages
from numpy import isin
import argparse
import sys
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import build_agent_graph
from app.agent.guardrails import GuardrailError, check_input, check_output
from app.config import settings
from app.core.logging import configure_agent_logging

def _message_text(content: str | list) -> str:
    if isinstance(content, str):
        return content

    return "".join(
        part if isinstance(part, str) else part.get("text", "")
        for part in content
        if isinstance(part, str)
        or isinstance(part, dict) and part.get("type") == "text"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="run the scikit-learn agent on one question",
    )
    parser.add_argument("question", help="Question for the agent")
    args = parser.parse_args()

    configure_agent_logging()

    try:
        check_input(args.question)
    except GuardrailError as exc:
        print(f"Запрос отклонён: {exc}", file=sys.stderr)
        return 1

    graph = build_agent_graph()
    thread_id = str(uuid.uuid4())
    started = time.perf_counter()

    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=args.question)],
                "iteration_count": 0,
            },
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 2 * settings.max_iterations + 2,
            },
        )
    except Exception as exc:
        print(
            f"Агент не завершил работу: {type(exc).__name__}",
            file=sys.stderr
        )
        return 1

    total_ms = int((time.perf_counter() - started) * 1000)
    messages = result["messages"]
    final = messages[-1] if messages else None

    if (
        not isinstance(final, AIMessage)
        or final.tool_calls
        or not _message_text(final.content).strip()
    ):
        print("Агент не сформировал итоговый ответ.", file=sys.stderr)
        return 1

    tools_used = [
        message.name
        for message in messages
        if isinstance(message, ToolMessage) and message.name
    ]

    answer, guardrail_reason = check_output(
        _message_text(final.content),
        tools_used,
    )
    termination_reason = result.get("termination_reason")

    print(f"\nОтвет:\n{answer}")
    print(f"\nИтераций: {result['iteration_count']}")
    print(f"Общее время: {total_ms} мс")
    print(f"Инструменты: {', '.join(tools_used) or 'не вызывались'}")

    if guardrail_reason:
        print(f"Проверка ответа: {guardrail_reason}")

    if termination_reason:
        print(f"Причина остановки: {termination_reason}")

    return 1 if termination_reason or guardrail_reason else 0


if __name__ == "__main__":
    raise SystemExit(main())