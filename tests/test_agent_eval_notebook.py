import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest


def recovery_helper(tmp_path, evaluate):
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks/agent_eval.ipynb").read_text(encoding="utf-8")
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name == "_recover_batch":
                namespace = {"json": json, "RUN_DIR": tmp_path, "_evaluate_batch": evaluate}
                exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook-recovery", "exec"), namespace)
                return namespace["_recover_batch"]
    pytest.fail("Notebook recovery helper is missing")


def report(*, error=False, stop=None):
    return {
        "complete": not error, "stop_reason": stop, "calls_attempted": 3,
        "questions": [{"id": "sh-01", "metrics": {
            "faithfulness": {"status": "error" if error else "completed",
                             "value": None if error else 0.8,
                             **({"error_type": "OpenAIAPIError"} if error else {})},
            "answer_relevancy": {"status": "completed", "value": 0.7},
        }}],
    }


def test_recovery_preserves_completed_metric_and_original(tmp_path):
    original = report(error=True)
    before = deepcopy(original)
    calls = []

    def evaluate(system, ids, name):
        calls.append((system, ids, name))
        result = report()
        result["questions"][0]["metrics"]["answer_relevancy"]["value"] = 0.99
        return result

    recovered = recovery_helper(tmp_path, evaluate)("baseline", "batch-1", original)
    assert recovered["complete"]
    assert recovered["questions"][0]["metrics"]["faithfulness"]["value"] == 0.8
    assert recovered["questions"][0]["metrics"]["answer_relevancy"]["value"] == 0.7
    assert len(calls) == 1 and calls[0][1] == ["sh-01"]
    assert original == before
    assert recovered["recovery_reports"]


def test_recovery_does_not_retry_completed_or_quota_limited_report(tmp_path):
    def evaluate(*args):
        pytest.fail("Unexpected judge retry")

    recover = recovery_helper(tmp_path, evaluate)
    assert recover("agent", "batch-1", report())["complete"]
    assert not recover("agent", "batch-1", report(error=True, stop="provider_limit"))["complete"]


def test_recovery_bounds_retries_and_keeps_failure(tmp_path):
    calls = []

    def evaluate(*args):
        calls.append(args)
        return report(error=True)

    recovered = recovery_helper(tmp_path, evaluate)("agent", "batch-1", report(error=True))
    assert not recovered["complete"]
    assert len(calls) == 2
    assert recovered["questions"][0]["metrics"]["faithfulness"]["value"] is None


def test_undefined_metric_is_not_retried_until_a_desired_score_appears(tmp_path):
    original = report(error=True)
    original["questions"][0]["metrics"]["faithfulness"] = {"status": "undefined", "value": None}

    def evaluate(*args):
        pytest.fail("Undefined metric must remain visible")

    recovered = recovery_helper(tmp_path, evaluate)("agent", "batch-1", original)
    assert not recovered["complete"]
