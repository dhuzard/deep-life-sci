"""Deterministic PMID citation checks.

`citations_exist` asks whether every cited PMID exists in the persistent abstract cache.
That catches invented identifiers cheaply, but the cache is shared across runs: a paper
fetched yesterday can make today's citation look grounded even when today's agent never
read it.

`citations_retrieved_this_run` closes that narrower provenance gap. It reads the
run-scoped source trace and requires every cited PMID to have had source content retrieved
in this run. PubMed search hits alone do not qualify; the trace summary includes PMIDs
only after abstract retrieval or access through linked PMC content.

Both checks are boolean and all-or-nothing. Neither proves that the source supports the
claim attached to it — that is a separate semantic-grounding problem.
"""

from __future__ import annotations

import re

from deep_life_sci.paths import ABSTRACT_CACHE
from evals.evaluators._guard import scores_only_completed_runs

# PubMed ids are 1-8 digits, but a bare 4-digit run in prose is almost always a year and
# a bare 2-3 digit one is a sample size. Requiring 7+ digits, or an explicit `PMID:`
# label, is what keeps the false-positive rate near zero on text full of numbers.
_LABELLED = re.compile(r"PMID[:\s]*(\d{1,8})", re.IGNORECASE)
_BARE = re.compile(r"\b(\d{7,8})\b")


def cited_pmids(text: str) -> set[str]:
    """Every PMID the answer appears to cite."""
    return {m for m in _LABELLED.findall(text)} | {m for m in _BARE.findall(text)}


@scores_only_completed_runs("citations_exist")
def citations_exist(run, example) -> dict:
    """True when every cited PMID is present in the host-side abstract cache."""
    answer = (run.outputs or {}).get("answer", "")
    pmids = cited_pmids(answer)

    if not pmids:
        # Not automatically a failure: the metadata-only questions ("which journals
        # publish the most...") are answerable without citing a single paper. Scored
        # None so it is excluded from the aggregate rather than dragging it down.
        return {
            "key": "citations_exist",
            "score": None,
            "comment": "no PMIDs cited — not applicable to this answer",
        }

    found = {p for p in pmids if (ABSTRACT_CACHE / f"{p}.json").exists()}
    missing = sorted(pmids - found)
    return {
        "key": "citations_exist",
        "score": not missing,
        "comment": (
            f"{len(found)}/{len(pmids)} cited PMIDs in the fetch cache"
            + (f"; unverifiable: {missing[:10]}" if missing else "")
        ),
    }


@scores_only_completed_runs("citations_retrieved_this_run")
def citations_retrieved_this_run(run, example) -> dict:
    """True when every cited PMID had source content retrieved in this run."""
    outputs = run.outputs or {}
    pmids = cited_pmids(outputs.get("answer", ""))

    if not pmids:
        return {
            "key": "citations_retrieved_this_run",
            "score": None,
            "comment": "no PMIDs cited — not applicable to this answer",
        }

    trace = outputs.get("source_trace")
    traced_pmids = trace.get("pmids") if isinstance(trace, dict) else None
    if not isinstance(traced_pmids, list):
        # Old experiments and callers that predate run provenance are unscoreable here.
        # Treating missing instrumentation as a citation failure would turn a harness
        # version difference into an apparent research-quality regression.
        return {
            "key": "citations_retrieved_this_run",
            "score": None,
            "comment": "run did not expose source_trace.pmids — cannot verify this metric",
        }

    retrieved = {str(pmid).strip() for pmid in traced_pmids if str(pmid).strip()}
    found = pmids & retrieved
    missing = sorted(pmids - retrieved)
    return {
        "key": "citations_retrieved_this_run",
        "score": not missing,
        "comment": (
            f"{len(found)}/{len(pmids)} cited PMIDs had source content retrieved in this run"
            + (f"; not retrieved: {missing[:10]}" if missing else "")
        ),
    }
