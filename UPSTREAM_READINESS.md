# Upstream contribution readiness

This fork uses the following review gate before a change is described as ready for a contribution to `langchain-samples/deep-life-sci`.

## Scope
- [ ] One coherent issue or measured limitation.
- [ ] Minimal diff that preserves Deep Life Sci architecture and scope.
- [ ] Current upstream main reviewed before finalization.
- [ ] Compatibility and migration implications documented.
- [ ] No unrelated refactor bundled.

## Review
- [ ] Complete diff read end to end.
- [ ] Applicable guidance in `AGENTS.md` and module guidance checked.
- [ ] Error paths, concurrency/context isolation, cache behavior, source rate limits, and sandbox boundaries reviewed where relevant.
- [ ] Large scientific source payloads remain outside root context.

## Validation
- [ ] Focused tests run and inspected.
- [ ] `uv run --group test pytest` run for runtime changes.
- [ ] Runtime changes checked on Python 3.12 and 3.13, or the exception is documented.
- [ ] `uv run ruff check .` run where applicable.
- [ ] Relevant structural/live evals run, or a specific rationale explains why they are not informative.
- [ ] Actual traces, retrieved sources, artifacts, provenance, evaluator comments, and failure/skip/N/A outcomes inspected where applicable.
- [ ] Any regression is fixed and affected checks rerun.

## Evidence to preserve

For every candidate, record in the related issue or fork PR: upstream base commit; candidate commit; exact focused/full test and lint results; eval run links; manual inspection findings; warnings/skips/N/A outcomes; known limitations and regression risk.

A green workflow alone is not sufficient evidence of readiness.
