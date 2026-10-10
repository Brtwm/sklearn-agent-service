import re

from app.config import settings

_PROMPT_INJECTION_PATTERNS = [
    r"\bignore\s+(previous|all)\s+instructions\b",
    r"\byou\s+are\s+(now|a)\b",
    r"\bsystem\s+prompt\b",
    r"<\|im_(start|end)\|>",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PROMPT_INJECTION_PATTERNS]

_ALLOWED_CHARS = re.compile(
    r"[A-Za-zА-Яа-яЁё0-9\s.,?!()\-—:;'\"$%/+=\[\]*^<>_&|@#~`{}\\]+"
)

_DECLARATIVE_MARKERS = (
    "according to the documentation",
    "according to the docs",
    "as stated in the documentation",
    "as stated in the docs",
    "the documentation says",
    "согласно документации",
    "как указано в документации",
    "в документации сказано",
)

class GuardrailError(ValueError):
    pass


def check_input(question: str) -> None:
    if not (1 <= len(question) <= 500):
        raise GuardrailError(f"length out of range: {len(question)}")
    if not question.strip():
        raise GuardrailError("blank question")
    if not _ALLOWED_CHARS.fullmatch(question):
        raise GuardrailError("forbidden characters in question")
    for pattern in _COMPILED:
        if pattern.search(question):
            raise GuardrailError(f"prompt-injection pattern: {pattern.pattern}")


def check_output(answer: str, trace_tools: list[str]) -> tuple[str, str | None]:
    """Return (safe_answer, reason), prioritizing unsupported source claims."""
    reason = None
    lowered = answer.lower()
    has_declarative = any(marker in lowered for marker in _DECLARATIVE_MARKERS)
    used_doc_search = "documentation_search" in trace_tools

    if has_declarative and not used_doc_search:
        answer = (
            "I cannot make a documentation-grounded claim without consulting "
            "the corpus first. Please rephrase or ask a more specific question."
        )
        reason = "declarative_without_source"

    limit = settings.agent_max_output_chars
    if len(answer) > limit:
        suffix = "\n...(truncated)"[:limit]
        answer = answer[: limit - len(suffix)] + suffix
        if reason is None:
            reason = "output_too_long"

    return answer, reason
