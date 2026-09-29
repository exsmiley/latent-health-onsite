import { useState } from "react";
import { fetchChunk } from "../api";
import { useChunk } from "../useChunk";
import type { Citation } from "../types";
import { ChunkBody } from "./ChunkPopover";

export function Sources({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;
  return (
    <div className="sources">
      <div className="sources-label">Sources</div>
      <ol>
        {citations.map((c) => (
          <SourceItem key={`${c.n}-${c.chunk_id}`} c={c} />
        ))}
      </ol>
    </div>
  );
}

function SourceItem({ c }: { c: Citation }) {
  const [open, setOpen] = useState(false);
  const state = useChunk(open ? c.chunk_id : null);
  return (
    <li id={`source-${c.chunk_id}`} className={open ? "open" : ""}>
      <div className="source-row">
        <span className="chip static">{c.n}</span>
        <a className="source-title" href={c.url} target="_blank" rel="noreferrer">
          {c.title}
          {c.section && <span className="muted"> › {c.section}</span>}
        </a>
        <button
          type="button"
          className="link-btn"
          aria-expanded={open}
          // Prefetch on hover so the panel opens instantly.
          onMouseEnter={() => void fetchChunk(c.chunk_id).catch(() => {})}
          onClick={() => setOpen((o) => !o)}
        >
          {open ? "Hide passage" : "Show passage"}
        </button>
      </div>
      {open && (
        <div className="source-panel">
          <ChunkBody state={state} />
          <div className="muted small">chunk {c.chunk_id}</div>
        </div>
      )}
    </li>
  );
}
