import { useDevMode } from "../devMode";
import type { AssistantState } from "../trace";
import { Answer } from "./Answer";
import { ChunkPopover, usePopover } from "./ChunkPopover";
import { Sources } from "./Sources";
import { Trace } from "./Trace";

export function AssistantMessage({ state }: { state: AssistantState }) {
  const pop = usePopover();
  const dev = useDevMode();
  const running = state.phase === "running";
  const streaming = running && state.text.length > 0;

  return (
    <div className="msg assistant">
      {dev ? (
        <Trace state={state} />
      ) : (
        running &&
        !state.text && (
          <div className="trace-pending">
            <span className="spinner" /> Thinking…
          </div>
        )
      )}

      {state.text && (
        <Answer
          text={state.text}
          citations={state.citations}
          streaming={streaming}
          onChipEnter={pop.hoverOpen}
          onChipLeave={pop.hoverClose}
          onChipClick={pop.toggle}
        />
      )}

      {state.errors.map((m, i) => (
        <div key={i} className="error-box" role="alert">
          <strong>Error:</strong> {m}
        </div>
      ))}

      {state.phase === "stopped" && <div className="muted small stopped">Stopped.</div>}

      {state.citations && state.citations.length > 0 && <Sources citations={state.citations} />}

      {pop.target && (
        <ChunkPopover
          target={pop.target}
          onMouseEnter={pop.cancelClose}
          onMouseLeave={pop.hoverClose}
          onClose={pop.close}
        />
      )}
    </div>
  );
}
