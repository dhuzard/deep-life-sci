"""Run-local provenance for scientific source tools.

The root model orchestrates source calls inside QuickJS, below LangGraph middleware.
`progress.py` already wraps tool coroutines at that seam; provenance uses the same seam
so it observes the real source calls without copying source payloads into model context.

A trace is opt-in through `research_trace()`. Outside that context the wrappers return
exactly what their inner tools return and retain no state. `ContextVar` keeps concurrent
runner calls isolated, and the QuickJS host bridge preserves the calling context when it
schedules the wrapped coroutine back onto the event loop (the same property progress
events rely on).

Only compact identifiers, queries, URLs, and failure text are retained. Abstracts, full
texts, registry records, and web-search answer bodies are deliberately excluded.
"""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class SourceEvent:
    """One scientific source-tool call, reduced to evaluator-safe provenance."""

    source: str
    tool: str
    operation: str
    query: str | None = None
    requested_ids: tuple[str, ...] = ()
    returned_ids: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()
    success: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("requested_ids", "returned_ids", "urls"):
            value[key] = list(value[key])
        return value


@dataclass
class ResearchTrace:
    """Mutable collector owned by one runner invocation."""

    events: list[SourceEvent] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        pmids = _unique(
            event_id
            for event in self.events
            if event.source == "pubmed"
            for event_id in event.returned_ids
        )
        pmcids = _unique(
            event_id
            for event in self.events
            if event.source == "pmc"
            for event_id in event.returned_ids
        )
        nct_ids = _unique(
            event_id
            for event in self.events
            if event.source == "ctgov"
            for event_id in event.returned_ids
        )
        urls = _unique(url for event in self.events for url in event.urls)
        return {
            "events": [event.as_dict() for event in self.events],
            "pmids": pmids,
            "pmcids": pmcids,
            "nct_ids": nct_ids,
            "urls": urls,
        }


_CURRENT_TRACE: ContextVar[ResearchTrace | None] = ContextVar(
    "deep_life_sci_research_trace", default=None
)

_TOOL_META = {
    "pubmed_search": ("pubmed", "search"),
    "fetch_abstracts": ("pubmed", "fetch"),
    "pmc_locate": ("pmc", "locate"),
    "fetch_full_text": ("pmc", "read"),
    "fetch_figures": ("pmc", "stage_figure"),
    "fetch_supplementary": ("pmc", "stage_supplementary"),
    "ctgov_search": ("ctgov", "search"),
    "ctgov_fetch": ("ctgov", "fetch"),
    "web_search": ("web", "search"),
}


@contextmanager
def research_trace() -> Iterator[ResearchTrace]:
    """Collect source events in this execution context and reset it on exit."""
    trace = ResearchTrace()
    token = _CURRENT_TRACE.set(trace)
    try:
        yield trace
    finally:
        _CURRENT_TRACE.reset(token)


def with_source_trace(tools: Sequence[BaseTool]) -> list[BaseTool]:
    """Copies of source tools that record compact provenance when a trace is active."""
    return [_wrap(tool) for tool in tools]


def _wrap(tool: BaseTool) -> BaseTool:
    inner = getattr(tool, "coroutine", None)
    if inner is None:
        return tool

    @functools.wraps(inner)
    async def traced(*args: Any, **kwargs: Any) -> Any:
        trace = _CURRENT_TRACE.get()
        if trace is None:
            return await inner(*args, **kwargs)

        call_args = _arguments(inner, args, kwargs)
        try:
            result = await inner(*args, **kwargs)
        except BaseException as exc:
            _append(trace, _event(tool.name, call_args, None, exc))
            raise
        _append(trace, _event(tool.name, call_args, result, None))
        return result

    return tool.model_copy(update={"coroutine": traced})


def _arguments(func: Callable, args: tuple, kwargs: dict) -> dict:
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        return dict(bound.arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def _append(trace: ResearchTrace, event: SourceEvent | None) -> None:
    """Tracing is observational; malformed provenance must never fail the source call."""
    if event is None:
        return
    try:
        trace.events.append(event)
    except Exception:  # noqa: BLE001
        pass


def _event(
    tool_name: str,
    args: dict[str, Any],
    result: Any,
    exception: BaseException | None,
) -> SourceEvent | None:
    meta = _TOOL_META.get(tool_name)
    if meta is None:
        return None

    try:
        source, operation = meta
        error = str(exception) if exception is not None else _result_error(tool_name, result)
        return SourceEvent(
            source=source,
            tool=tool_name,
            operation=operation,
            query=_query(tool_name, args, result),
            requested_ids=_requested_ids(tool_name, args),
            returned_ids=_returned_ids(tool_name, result),
            urls=_urls(tool_name, result),
            success=error is None,
            error=error,
        )
    except Exception:  # noqa: BLE001
        return SourceEvent(
            source=meta[0],
            tool=tool_name,
            operation=meta[1],
            success=False,
            error="provenance extraction failed",
        )


def _query(tool_name: str, args: dict[str, Any], result: Any) -> str | None:
    if tool_name == "pubmed_search":
        return _text(args.get("term"))
    if tool_name == "web_search":
        return _text(args.get("query"))
    if tool_name == "ctgov_search":
        sent = result.get("query_sent") if isinstance(result, dict) else None
        if isinstance(sent, dict):
            return json.dumps(sent, sort_keys=True, separators=(",", ":"))
        selected = {
            key: value
            for key, value in args.items()
            if key
            in {
                "condition",
                "intervention",
                "term",
                "title",
                "sponsor",
                "status",
                "filter_advanced",
            }
            and value
        }
        return json.dumps(selected, sort_keys=True, separators=(",", ":")) if selected else None
    return None


def _requested_ids(tool_name: str, args: dict[str, Any]) -> tuple[str, ...]:
    key = {
        "fetch_abstracts": "pmids",
        "pmc_locate": "pmcids",
        "fetch_full_text": "pmcids",
        "ctgov_fetch": "nct_ids",
    }.get(tool_name)
    if key is not None:
        return _ids(args.get(key))
    if tool_name in {"fetch_figures", "fetch_supplementary"}:
        return _ids(args.get("pmcid"))
    return ()


def _returned_ids(tool_name: str, result: Any) -> tuple[str, ...]:
    if not isinstance(result, dict):
        return ()
    if tool_name in {"pubmed_search", "ctgov_search"}:
        field = "pmid" if tool_name == "pubmed_search" else "nct_id"
        records = result.get("records")
        if isinstance(records, list):
            return _ids(
                record.get(field)
                for record in records
                if isinstance(record, dict)
            )
    if tool_name in {"fetch_abstracts", "fetch_full_text", "ctgov_fetch"}:
        records = result.get("records")
        return _ids(records.keys()) if isinstance(records, dict) else ()
    if tool_name == "pmc_locate":
        available = result.get("available")
        return _ids(available.keys()) if isinstance(available, dict) else ()
    if tool_name in {"fetch_figures", "fetch_supplementary"}:
        staged = result.get("staged")
        if isinstance(staged, list):
            return _ids(
                item.get("pmcid")
                for item in staged
                if isinstance(item, dict)
            )
    return ()


def _urls(tool_name: str, result: Any) -> tuple[str, ...]:
    if tool_name != "web_search" or not isinstance(result, dict):
        return ()
    sources = result.get("sources")
    if not isinstance(sources, list):
        return ()
    return _unique_tuple(
        source.get("url")
        for source in sources
        if isinstance(source, dict)
    )


def _result_error(tool_name: str, result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    if error := result.get("error"):
        return _text(error)
    if tool_name == "web_search" and not result.get("searched"):
        warnings = result.get("warnings")
        if isinstance(warnings, list) and warnings:
            return _text(warnings[0])
    return None


def _ids(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, int)):
        values = [values]
    try:
        return _unique_tuple(values)
    except TypeError:
        return ()


def _unique(values) -> list[str]:
    return list(_unique_tuple(values))


def _unique_tuple(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(text for value in values if (text := _text(value))))


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
