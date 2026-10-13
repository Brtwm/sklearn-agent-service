# Agent vs baseline RAG — noreason-v2

RAGAS complete: True

| Metric | Baseline RAG | Agent | N paired |
| --- | --- | --- | --- |
| faithfulness (documentation) | 0.926 | 0.470 | 9 |
| answer_relevancy (documentation) | 0.940 | 0.942 | 9 |
| Numerical accuracy (multi-hop) | 1.000 | 1.000 | 5 |
| Math/ML answer accuracy (multi-hop) | 1.000 | 0.800 | 5 |
| Task success (multi-hop) | — | 0.800 | 5 |
| Mean HTTP time, ms (all 15) | 3981.600 | 8546.600 | 15 |
| Mean agent iterations (all 15) | — | 2.267 | 15 |
| Truncated answers (all 15) | — | 5.000 | 15 |

## Interpretation limits

- RAGAS covers nine documentation questions; multi-hop is reviewed separately on five cases.
- Each answer is judged against its original context; retrieved contexts may differ between systems.
- Same-model judging, strictness=1 and one run limit confidence.
- Faithfulness does not validate individual citations, tool execution, language or all facts.
- Baseline execution-like wording is excluded from mathematical accuracy; see manual_reviews.json.
- Five agent single-hop answers were truncated; delivered answers remain in the evaluation.
- The agent F1 number is correct, but the claim that it is lower than both 0.8 and 0.6 is false.
- The capabilities question is not included in documentation grounding.
- One HTTP 502 was recovered with a question-level retry; original failed and retry reports remain saved.
- Only the first completed score per metric is used; source_report identifies its origin.

## Observed results

Baseline Faithfulness is about 0.926, agent about 0.470; relevancy is about 0.940 vs 0.942.
The agent adds more examples and general explanations not established by its retrieved chunks.
For sh-07 the judge marked application-specific cost examples unsupported; for sh-02 the retrieved alpha context lacked several claimed Ridge details.
Low document support does not establish factual error: valid illustrative calculations and general knowledge may be penalized too.
This judge's statement extraction and verdicts can be imperfect; inspect the raw reports before treating a score as an objective truth.
