import { useEffect, useRef, useState } from "react";
import {
  continuation,
  formatArgs,
  OUTCOME_LABEL,
  summaryLine,
  type AssistantState,
  type TurnState,
} from "../trace";
import type { OutcomeData, ResearchAnswerData, Stage, StatusData, Verdict } from "../types";

const STAGES: { key: Stage; label: string }[] = [
  { key: "research", label: "Research" },
  { key: "evaluate", label: "Evaluate" },
  { key: "respond", label: "Respond" },
];

const turnOf = (turn: number, max: number | null) => (max != null ? `Turn ${turn} / ${max}` : `Turn ${turn}`);

export function Trace({ state }: { state: AssistantState }) {
  const running = state.phase === "running";
  const [open, setOpen] = useState(running);
  const wasRunning = useRef(running);

  // Live while running; auto-collapse once when the run finishes.
  useEffect(() => {
    if (wasRunning.current && !running) setOpen(false);
    wasRunning.current = running;
  }, [running]);

  if (state.turns.length === 0 && !state.status) {
    return running ? (
      <div className="trace-pending">
        <span className="spinner" /> Starting…
      </div>
    ) : null;
  }

  const lastTurn = state.turns[state.turns.length - 1]?.turn;

  return (
    <details
      className={`trace${running ? " live" : ""}`}
      open={open}
      onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}
    >
      <summary>
        <span className="caret" aria-hidden>
          ▸
        </span>
        <span className="trace-title">Research trace</span>
        {running && state.status ? (
          <span className="trace-live">
            <span className="spinner" />
            <StageSteps stage={state.status.stage} />
            {state.status.turn != null && (
              <span className="muted trace-where">{turnOf(state.status.turn, state.maxTurns)}</span>
            )}
          </span>
        ) : (
          <span className="muted trace-summary">{summaryLine(state)}</span>
        )}
      </summary>

      <div className="trace-body">
        {running && state.status?.message && <div className="trace-status">{state.status.message}</div>}
        <ol className="timeline">
          {state.turns.map((t) => (
            <TurnView
              key={t.turn}
              t={t}
              maxTurns={state.maxTurns}
              running={running}
              isLast={t.turn === lastTurn}
              status={state.status}
            />
          ))}
        </ol>
        {state.outcome && <OutcomeLine outcome={state.outcome} maxTurns={state.maxTurns} />}
      </div>
    </details>
  );
}

function StageSteps({ stage }: { stage: Stage }) {
  const idx = STAGES.findIndex((s) => s.key === stage);
  return (
    <span className="stages">
      {STAGES.map((s, i) => (
        <span key={s.key} className={`stage${i === idx ? " active" : i < idx ? " past" : ""}`}>
          {s.label}
        </span>
      ))}
    </span>
  );
}

function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return (
    <span className={`badge ${verdict}`}>{verdict === "supported" ? "✓ Supported" : "✗ Unsupported"}</span>
  );
}

function TurnView({
  t,
  maxTurns,
  running,
  isLast,
  status,
}: {
  t: TurnState;
  maxTurns: number | null;
  running: boolean;
  isLast: boolean;
  status: StatusData | null;
}) {
  const live = running && isLast;
  const thinking = live && status?.stage === "research" && t.calls.length === 0 && !t.answer;
  const evaluating =
    running && status?.stage === "evaluate" && status.turn === t.turn && t.answer?.status === "answered" && !t.evaluation;
  const next = continuation(t, maxTurns);
  const tone = t.evaluation?.verdict ?? (t.answer?.status === "invalid" ? "unsupported" : "");

  return (
    <li className={`turn${tone ? ` ${tone}` : ""}`}>
      <div className="turn-label">{turnOf(t.turn, maxTurns)}</div>

      {thinking && (
        <div className="muted small">
          <span className="spinner" /> thinking…
        </div>
      )}

      {t.calls.length > 0 && (
        <ul className="calls">
          {t.calls.map(({ call, result }) => (
            <li key={call.id} className="call">
              <span className={`tool-name tool-${call.name}`}>{call.name}</span>
              <span className="call-args">{formatArgs(call.arguments)}</span>
              <span className="call-result">
                {result ? (
                  <>→ {result.summary}</>
                ) : running ? (
                  <span className="spinner" />
                ) : (
                  <span className="muted">no result</span>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}

      {t.answer && <AnswerView a={t.answer} />}

      {evaluating && (
        <div className="evaluation pending">
          <span className="step-label">Evaluator</span> <span className="spinner" /> checking cited chunks…
        </div>
      )}

      {t.evaluation && (
        <div className={`evaluation ${t.evaluation.verdict}`}>
          <div className="eval-head">
            <span className="step-label">Evaluator</span>
            <VerdictBadge verdict={t.evaluation.verdict} />
          </div>
          {t.evaluation.feedback && <p>{t.evaluation.feedback}</p>}
          {t.evaluation.independent_answer && (
            <details className="independent">
              <summary>Evaluator's independent answer</summary>
              <p>{t.evaluation.independent_answer}</p>
            </details>
          )}
        </div>
      )}

      {next?.kind === "continue" && (
        <div className="continue">
          ↓ continuing research
          {next.turnsLeft != null && ` (${next.turnsLeft} turn${next.turnsLeft === 1 ? "" : "s"} left)`}
          {t.evaluation ? " with the evaluator's feedback" : ""}
        </div>
      )}
      {next?.kind === "exhausted" && <div className="continue exhausted">No turns left</div>}
    </li>
  );
}

function AnswerView({ a }: { a: ResearchAnswerData }) {
  if (a.status === "answered") {
    return (
      <div className="proposed">
        <div className="step-label">Answer</div>
        <p>{a.answer}</p>
        <div className="muted small">cites chunks {a.citations.join(", ") || "none"}</div>
      </div>
    );
  }
  if (a.status === "not_found") {
    return (
      <div className="proposed not-found">
        <div className="step-label">Not found</div>
        {a.reason && <p>{a.reason}</p>}
      </div>
    );
  }
  return (
    <div className="proposed invalid">
      <div className="step-label">Invalid answer</div>
      <p className="error-text">{a.reason || "The answer was rejected."}</p>
    </div>
  );
}

function OutcomeLine({ outcome, maxTurns }: { outcome: OutcomeData; maxTurns: number | null }) {
  const n = outcome.turns_used;
  const used = maxTurns != null ? `${n} of ${maxTurns} turns` : `${n} turn${n === 1 ? "" : "s"}`;
  return (
    <div className={`outcome ${outcome.result}`}>
      <span className={`badge ${outcome.result}`}>{OUTCOME_LABEL[outcome.result]}</span>
      <span className="muted">{outcome.result === "out_of_turns" ? `all ${n} turns used` : `after ${used}`}</span>
    </div>
  );
}
