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
    "outcome",
    "citations",
    "token",
    "done",
    "error",
]
Stage = Literal["research", "evaluate", "respond"]
AnswerStatus = Literal["answered", "not_found", "invalid"]
OutcomeResult = Literal["supported", "not_found", "out_of_turns"]


class Event(BaseModel):
    type: EventType
    data: Any

    def to_sse(self) -> str:
        payload = json.dumps(self.data, ensure_ascii=False, separators=(",", ":"))
        return f"event: {self.type}\ndata: {payload}\n\n"


def status(stage: Stage, turn: int | None, max_turns: int, message: str) -> Event:
    return Event(
        type="status",
        data={"stage": stage, "turn": turn, "max_turns": max_turns, "message": message},
    )


def tool_call(id: str, name: str, arguments: Any, turn: int) -> Event:
    return Event(
        type="tool_call", data={"id": id, "name": name, "arguments": arguments, "turn": turn}
    )


def tool_result(id: str, name: str, summary: str) -> Event:
    return Event(type="tool_result", data={"id": id, "name": name, "summary": summary})


def research_answer(
    turn: int, status: AnswerStatus, answer: str, citations: list[int], reason: str
) -> Event:
    return Event(
        type="research_answer",
        data={
            "turn": turn,
            "status": status,
            "answer": answer,
            "citations": citations,
            "reason": reason,
        },
    )


def evaluation(turn: int, verdict: str, independent_answer: str, feedback: str) -> Event:
    return Event(
        type="evaluation",
        data={
            "turn": turn,
            "verdict": verdict,
            "independent_answer": independent_answer,
            "feedback": feedback,
        },
    )


def outcome(result: OutcomeResult, turns_used: int) -> Event:
    return Event(type="outcome", data={"result": result, "turns_used": turns_used})


def citations(items: list[dict]) -> Event:
    return Event(type="citations", data=items)


def token(delta: str) -> Event:
    return Event(type="token", data={"delta": delta})


def done() -> Event:
    return Event(type="done", data={})


def error(message: str) -> Event:
    return Event(type="error", data={"message": message})
