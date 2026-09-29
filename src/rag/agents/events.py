"""Pipeline events. These are exactly the SSE events in docs/ARCHITECTURE.md ("HTTP / SSE contract")."""

import json
from typing import Any, Literal

from pydantic import BaseModel

EventType = Literal[
    "status",
    "tool_call",
    "tool_result",
    "research_answer",
    "evaluation",
    "citations",
    "token",
    "done",
    "error",
]
Stage = Literal["research", "evaluate", "respond"]


class Event(BaseModel):
    type: EventType
    data: Any

    def to_sse(self) -> str:
        payload = json.dumps(self.data, ensure_ascii=False, separators=(",", ":"))
        return f"event: {self.type}\ndata: {payload}\n\n"


def status(stage: Stage, round: int, turn: int | None, message: str) -> Event:
    return Event(
        type="status", data={"stage": stage, "round": round, "turn": turn, "message": message}
    )


def tool_call(id: str, name: str, arguments: Any, round: int, turn: int) -> Event:
    return Event(
        type="tool_call",
        data={"id": id, "name": name, "arguments": arguments, "round": round, "turn": turn},
    )


def tool_result(id: str, name: str, summary: str) -> Event:
    return Event(type="tool_result", data={"id": id, "name": name, "summary": summary})


def research_answer(round: int, answer: str, citations: list[int]) -> Event:
    return Event(
        type="research_answer", data={"round": round, "answer": answer, "citations": citations}
    )


def evaluation(round: int, verdict: str, independent_answer: str, feedback: str) -> Event:
    return Event(
        type="evaluation",
        data={
            "round": round,
            "verdict": verdict,
            "independent_answer": independent_answer,
            "feedback": feedback,
        },
    )


def citations(items: list[dict]) -> Event:
    return Event(type="citations", data=items)


def token(delta: str) -> Event:
    return Event(type="token", data={"delta": delta})


def done() -> Event:
    return Event(type="done", data={})


def error(message: str) -> Event:
    return Event(type="error", data={"message": message})
