import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useChunk } from "../useChunk";
import type { Citation } from "../types";

export interface PopoverTarget {
  citation: Citation;
  anchor: HTMLElement;
  pinned: boolean;
}

/**
 * Hover/click popover controller shared by all chips of one message.
 * Hover opens after a short delay and closes when the pointer leaves both chip and popover;
 * click pins it open until clicking elsewhere or pressing Escape.
 */
export function usePopover() {
  const [target, setTarget] = useState<PopoverTarget | null>(null);
  const openTimer = useRef<number | undefined>(undefined);
  const closeTimer = useRef<number | undefined>(undefined);

  const clearTimers = () => {
    window.clearTimeout(openTimer.current);
    window.clearTimeout(closeTimer.current);
  };

  const hoverOpen = useCallback((citation: Citation, anchor: HTMLElement) => {
    clearTimers();
    openTimer.current = window.setTimeout(() => {
      setTarget((cur) => (cur?.pinned ? cur : { citation, anchor, pinned: false }));
    }, 150);
  }, []);

  const hoverClose = useCallback(() => {
    window.clearTimeout(openTimer.current);
    closeTimer.current = window.setTimeout(() => {
      setTarget((cur) => (cur?.pinned ? cur : null));
    }, 200);
  }, []);

  const cancelClose = useCallback(() => window.clearTimeout(closeTimer.current), []);

  const toggle = useCallback((citation: Citation, anchor: HTMLElement) => {
    clearTimers();
    setTarget((cur) =>
      cur?.pinned && cur.citation.n === citation.n ? null : { citation, anchor, pinned: true },
    );
  }, []);

  const close = useCallback(() => {
    clearTimers();
    setTarget(null);
  }, []);

  useEffect(() => () => clearTimers(), []);

  return { target, hoverOpen, hoverClose, cancelClose, toggle, close };
}

interface Props {
  target: PopoverTarget;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
  onClose: () => void;
}

export function ChunkPopover({ target, onMouseEnter, onMouseLeave, onClose }: Props) {
  const { citation, anchor, pinned } = target;
  const state = useChunk(citation.chunk_id);
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number; above: boolean } | null>(null);

  const place = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const a = anchor.getBoundingClientRect();
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    const margin = 8;
    let left = a.left + a.width / 2 - w / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - w - margin));
    const below = a.bottom + 6;
    const above = below + h > window.innerHeight - margin && a.top - 6 - h > margin;
    setPos({ top: above ? a.top - 6 - h : below, left, above });
  }, [anchor]);

  useLayoutEffect(place, [place, state.status]);

  useEffect(() => {
    const onScroll = () => place();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (pinned && !ref.current?.contains(t) && !anchor.contains(t)) onClose();
    };
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onScroll);
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    return () => {
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onScroll);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown);
    };
  }, [place, onClose, pinned, anchor]);

  return (
    <div
      ref={ref}
      className="popover"
      role="dialog"
      aria-label={`Source ${citation.n}`}
      style={pos ? { top: pos.top, left: pos.left } : { visibility: "hidden", top: 0, left: 0 }}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="popover-head">
        <span className="chip static">{citation.n}</span>
        <span className="popover-title">
          {citation.title}
          {citation.section && <span className="muted"> › {citation.section}</span>}
        </span>
        {pinned && (
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            ×
          </button>
        )}
      </div>
      <ChunkBody state={state} />
      <a className="popover-link" href={citation.url} target="_blank" rel="noreferrer">
        Open on Wikipedia ↗
      </a>
    </div>
  );
}

export function ChunkBody({ state }: { state: ReturnType<typeof useChunk> }) {
  if (state.status === "loading" || state.status === "idle")
    return <div className="chunk-text muted">Loading passage…</div>;
  if (state.status === "error") return <div className="chunk-text error-text">{state.message}</div>;
  return <div className="chunk-text">{state.chunk.text}</div>;
}
