import { useEffect, useRef, useState } from "react";
import { formatArgs, type AssistantState, type RoundState } from "../trace";
import type { Stage } from "../types";

const STAGES: { key: Stage; label: string }[] = [
  { key: "research", label: "Research" },
  { key: "evaluate", label: "Evaluate" },
  { key: "respond", label: "Respond" },
];

export function Trace({ state }: { state: AssistantState }) {
  const running = state.phase === "running";
  const [open, setOpen] = useState(running);
  const wasRunning = useRef(running);

  // Live while running; auto-collapse once when the run finishes.
  useEffect(() => {
    if (wasRunning.current && !running) setOpen(false);
    wasRunning.current = running;
  }, [running]);

  if (state.rounds.length === 0 && !state.status) {
    return running ? (
      <div className="trace-pending">
        <span className="spinner" /> Starting…
      </div>
    ) : null;
  }

  const toolCount = state.rounds.reduce(
    (n, r) => n + r.turns.reduce((m, t) => m + t.calls.length, 0),
    0,
  );
  const last = state.rounds[state.rounds.length - 1];
  const verdict = last?.evaluation?.verdict;

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
            <span className="muted trace-where">
              Round {state.status.round}
              {state.status.turn != null && ` · turn ${state.status.turn}`}
            </span>
          </span>
        ) : (
          <span className="muted trace-summary">
            {state.rounds.length} round{state.rounds.length === 1 ? "" : "s"} · {toolCount} tool call
            {toolCount === 1 ? "" : "s"}
            {verdict && <VerdictBadge verdict={verdict} small />}
          </span>
        )}
      </summary>

      <div className="trace-body">
        {running && state.status?.message && <div className="trace-status">{state.status.message}</div>}
        <ol className="rounds">
          {state.rounds.map((r, i) => (
            <RoundView
              key={r.round}
              r={r}
              isCurrent={running && i === state.rounds.length - 1}
              hasNext={i < state.rounds.length - 1}
              currentStage={state.status?.round === r.round ? state.status.stage : undefined}
            />
          ))}
        </ol>
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

function VerdictBadge({ verdict, small }: { verdict: "supported" | "unsupported"; small?: boolean }) {
  return (
    <span className={`badge ${verdict}${small ? " small" : ""}`}>
      {verdict === "supported" ? "✓ Supported" : "✗ Unsupported"}
    </span>
  );
}

function RoundView({
  r,
  isCurrent,
  hasNext,
  currentStage,
}: {
  r: RoundState;
  isCurrent: boolean;
  hasNext: boolean;
  currentStage?: Stage;
}) {
  const evaluating = isCurrent && currentStage === "evaluate" && !r.evaluation;
  return (
    <li className={`round${r.evaluation ? ` ${r.evaluation.verdict}` : ""}`}>
      <div className="round-head">
        <span className="round-label">Round {r.round}</span>
        {r.round > 1 && <span className="tag">retry</span>}
      </div>

      {r.turns.map((t) => (
        <div className="turn" key={t.turn}>
          <div className="turn-label">Turn {t.turn}</div>
          {t.calls.length === 0 && isCurrent && currentStage === "research" && (
            <div className="muted small">
              <span className="spinner" /> thinking…
            </div>
          )}
          <ul className="calls">
            {t.calls.map(({ call, result }) => (
              <li key={call.id} className="call">
                <span className={`tool-name tool-${call.name}`}>{call.name}</span>
                <span className="call-args">{formatArgs(call.name, call.arguments)}</span>
                <span className="call-result">
                  {result ? (
                    <>→ {result.summary}</>
                  ) : isCurrent ? (
                    <span className="spinner" />
                  ) : (
                    <span className="muted">no result</span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ))}

      {r.answer && (
        <div className="proposed">
          <div className="step-label">Proposed answer</div>
          <p>{r.answer.answer}</p>
          <div className="muted small">cites chunks {r.answer.citations.join(", ") || "none"}</div>
        </div>
      )}

      {evaluating && (
        <div className="evaluation pending">
          <span className="step-label">Evaluator</span> <span className="spinner" /> checking cited chunks…
        </div>
      )}

      {r.evaluation && (
        <div className={`evaluation ${r.evaluation.verdict}`}>
          <div className="eval-head">
            <span className="step-label">Evaluator</span>
            <VerdictBadge verdict={r.evaluation.verdict} />
          </div>
          {r.evaluation.feedback && <p>{r.evaluation.feedback}</p>}
          {r.evaluation.independent_answer && (
            <details className="independent">
              <summary>Evaluator's independent answer</summary>
              <p>{r.evaluation.independent_answer}</p>
            </details>
          )}
        </div>
      )}

      {r.evaluation?.verdict === "unsupported" &&
        (hasNext ? (
          <div className="retry-arrow">↻ back to research with the evaluator's feedback</div>
        ) : (
          currentStage === "respond" && (
            <div className="retry-arrow final">Max rounds reached: answer not found</div>
          )
        ))}
    </li>
  );
}
