"""Tool definitions (OpenAI Responses API, strict mode) and the dispatcher the agents call."""

import json

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from rag.config import get_settings
from rag.tools.fetch import fetch
from rag.tools.keyword import keyword_search
from rag.tools.models import FetchResult, QueryResults
from rag.tools.search import semantic_search

MAX_QUERIES = 5
MAX_TOP_K = 10
MAX_FETCH_IDS = 50

_TOP_K_SCHEMA = {
    "type": ["integer", "null"],
    "description": (
        f"Hits per query, 1-10. Pass null for the default ({get_settings().search_top_k})."
    ),
}


def _queries_schema(desc: str) -> dict:
    return {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": MAX_QUERIES,
        "description": desc,
    }


TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "name": "semantic_search",
        "description": (
            "Meaning-based search over Simple English Wikipedia, split into paragraph-aligned "
            "chunks. Takes 1-5 natural-language queries that run in parallel; each returns its "
            "top-k chunks as {chunk_id, article_id, title, section, score, blurb}, where blurb is "
            "only the first ~40 words of the chunk. Send several queries at once: different "
            "phrasings of the question, and separate sub-questions for multi-part or multi-hop "
            "questions (e.g. one per entity). Blurbs are previews: fetch the chunk_ids you need "
            "to read the full text before relying on or citing them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": _queries_schema(
                    "1-5 natural-language queries, e.g. questions or descriptive statements."
                ),
                "top_k": dict(_TOP_K_SCHEMA),
            },
            "required": ["queries", "top_k"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "keyword_search",
        "description": (
            "Exact-term full-text search (Postgres, English stemming) over the same chunks, "
            "ranked by term density. Best for names, dates, numbers, rare words and exact "
            "phrases that meaning-based search may blur. Takes 1-5 queries run in parallel, "
            'using web-search syntax: "quoted phrase" for exact phrases, OR between '
            "alternatives, -word to exclude; plain words are ANDed, so keep queries short "
            "(2-4 key terms). Returns the same hit shape as semantic_search. Queries made only "
            "of stop words return no hits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": _queries_schema(
                    'Keyword queries, e.g. "Treaty of Versailles" 1919, or Lincoln -Nebraska.'
                ),
                "top_k": dict(_TOP_K_SCHEMA),
            },
            "required": ["queries", "top_k"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "fetch",
        "description": (
            "Read full text. Prefer chunk_ids: returns each chunk's full text with its title, "
            "section and url. article_ids returns whole articles, which can be very long; use "
            "them only when you truly need broad context. Citations must be chunk ids: the "
            "evaluator only sees the text of the chunks you cite and will not accept whole "
            "articles, so after reading an article, fetch and cite the specific chunk ids "
            "(listed in the article's chunk_ids, in order) that support your answer. Unknown "
            "ids are reported in missing_chunk_ids / missing_article_ids. Pass [] for the "
            "list you are not using."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Chunk ids to read in full (from search hits). [] for none.",
                },
                "article_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Article ids to read in full. Usually []; use sparingly.",
                },
            },
            "required": ["chunk_ids", "article_ids"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

TOOL_NAMES = frozenset(spec["name"] for spec in TOOL_SPECS)


# ---- argument validation -------------------------------------------------------------------


class _SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[str]
    top_k: int | None = None

    @field_validator("queries")
    @classmethod
    def _check_queries(cls, v: list[str]) -> list[str]:
        v = [q.strip() for q in v if q and q.strip()]
        if not v:
            raise ValueError("queries must contain at least one non-empty string")
        if len(v) > MAX_QUERIES:
            raise ValueError(f"at most {MAX_QUERIES} queries per call (got {len(v)})")
        return v

    def k(self) -> int:
        k = get_settings().search_top_k if self.top_k is None else self.top_k
        return min(max(k, 1), MAX_TOP_K)


class _FetchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_ids: list[int] = Field(default_factory=list)
    article_ids: list[int] = Field(default_factory=list)

    def check(self) -> None:
        if not self.chunk_ids and not self.article_ids:
            raise ValueError("provide at least one chunk_id or article_id")
        if len(self.chunk_ids) + len(self.article_ids) > MAX_FETCH_IDS:
            raise ValueError(f"at most {MAX_FETCH_IDS} ids per fetch call")


_RESULTS = TypeAdapter(list[QueryResults])


def _error(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False, separators=(",", ":"))


def _format_validation(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "arguments"
        parts.append(f"{loc}: {err['msg']}")
    return "invalid arguments: " + "; ".join(parts)


async def dispatch(name: str, args: dict | str) -> str:
    """Run a tool and return the compact JSON string handed back to the model.

    Never raises: unknown tools, invalid arguments and tool failures come back as
    '{"error": "..."}'.

    Shapes on success:
      semantic_search / keyword_search -> [{"query", "hits": [{"chunk_id", "article_id", "title",
                                           "section"?, "score", "blurb"}]}]  (section omitted if null)
      fetch -> {"chunks": [...], "articles": [...], "missing_chunk_ids": [...],
                "missing_article_ids": [...]}
    """
    if name not in TOOL_NAMES:
        return _error(f"unknown tool {name!r}; available: {', '.join(sorted(TOOL_NAMES))}")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as e:
            return _error(f"arguments are not valid JSON: {e}")
    if not isinstance(args, dict):
        return _error("arguments must be a JSON object")
    try:
        if name == "fetch":
            fa = _FetchArgs.model_validate(args)
            fa.check()
            result: FetchResult = await fetch(fa.chunk_ids, fa.article_ids)
            return result.model_dump_json(exclude_none=True)
        sa = _SearchArgs.model_validate(args)
        tool = semantic_search if name == "semantic_search" else keyword_search
        results = await tool(sa.queries, sa.k())
        return _RESULTS.dump_json(results, exclude_none=True).decode()
    except ValidationError as e:
        return _error(_format_validation(e))
    except ValueError as e:
        return _error(f"invalid arguments: {e}")
    except Exception as e:  # noqa: BLE001 - tool failure (DB, embeddings API) goes to the model
        return _error(f"{name} failed: {type(e).__name__}: {e}")


# ---- UI summaries --------------------------------------------------------------------------


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def summarize(name: str, result: str | list | dict | FetchResult) -> str:
    """One-line summary of a tool result for the UI, e.g. "3 queries, 15 hits".

    Accepts the JSON string from `dispatch`, its parsed form, or the tool's return value.
    """
    try:
        if isinstance(result, str):
            result = json.loads(result)
        if isinstance(result, FetchResult):
            result = result.model_dump()
        if isinstance(result, dict) and "error" in result:
            return f"error: {result['error']}"
        if name == "fetch" and isinstance(result, dict):
            n_c = len(result.get("chunks", []))
            n_a = len(result.get("articles", []))
            missing = len(result.get("missing_chunk_ids", [])) + len(
                result.get("missing_article_ids", [])
            )
            s = f"fetched {_plural(n_c, 'chunk')}, {_plural(n_a, 'article')}"
            return s + (f", {missing} missing" if missing else "")
        if isinstance(result, list):
            hits = 0
            for r in result:
                h = r.hits if isinstance(r, QueryResults) else r.get("hits", [])
                hits += len(h)
            return f"{_plural(len(result), 'query', 'queries')}, {_plural(hits, 'hit')}"
    except (ValueError, TypeError, AttributeError):
        return f"{name} done"
    return f"{name} done"
