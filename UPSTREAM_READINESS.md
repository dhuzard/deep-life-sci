# Upstream contribution readiness

This fork uses the following gate before a change is described as ready for contribution to `langchain-samples/deep-life-sci`. The label `PR-ready` is reserved for work that has passed every applicable check below and has preserved the supporting evidence.

## 1. Scope
- [ ] One coherent issue or measured limitation.
- [ ] Current upstream `main` reviewed before finalization.
- [ ] Minimal diff that preserves Deep Life Sci architecture and identity.
- [ ] Backwards-compatibility and migration implications documented.
- [ ] No unrelated refactor or speculative abstraction bundled.

## 2. Code review
- [ ] Complete diff read end to end.
- [ ] Applicable guidance in `AGENTS.md`, `tests/test_invariants.py`, and module-specific guidance checked.
- [ ] Error paths, concurrency/context isolation, cache behavior, source rate limits, sandbox boundaries, and tool-return contracts reviewed where relevant.
- [ ] Large scientific source payloads remain outside root context.
- [ ] The final diff is reviewed again after the last fix.

## 3. Validation
- [ ] Focused tests for the changed area run and inspected.
- [ ] Runtime changes: `uv run --group test pytest`.
- [ ] Runtime changes: Python 3.12 and 3.13, or the exception is documented.
- [ ] `uv run ruff check .` where applicable.
- [ ] Relevant structural/live evals run, or a specific rationale explains why they are not informative.
- [ ] Actual traces, retrieved sources, artifacts, provenance, evaluator comments, failure/skip/N/A outcomes, and context/tool behavior inspected where applicable.
- [ ] Any regression is fixed and affected checks rerun.

A green command or workflow is necessary evidence, not sufficient evidence.

## 4. Evidence to preserve

Record in the related issue or fork PR:

- upstream base commit and candidate commit;
- exact focused/full test and lint results;
- eval run links and the evaluated seed(s);
- manual inspection findings, including failures and non-obvious passes;
- warnings, skips, N/A outcomes, and external-credential limitations;
- known limitations, regression risk, and dependency/stacking notes.

## 5. Proposed upstream PR

Only after the gate passes, prepare a narrow proposal containing:

- a concise title;
- motivation tied to an existing or measured Deep Life Sci limitation;
- implementation scope and important unchanged behavior;
- exact validation/evaluation evidence;
- known limitations and regression risk.

Do not open an upstream PR solely because implementation or CI is green.
