import { memo, useMemo } from "react";
import Markdown, { type Components } from "react-markdown";
import { CITE_PREFIX, linkifyCitations } from "../citations";
import type { Citation } from "../types";

interface Props {
  text: string;
  citations: Citation[] | null;
  streaming: boolean;
  onChipEnter: (c: Citation, el: HTMLElement) => void;
  onChipLeave: () => void;
  onChipClick: (c: Citation, el: HTMLElement) => void;
}

export const Answer = memo(function Answer({
  text,
  citations,
  streaming,
  onChipEnter,
  onChipLeave,
  onChipClick,
}: Props) {
  const byN = useMemo(() => new Map((citations ?? []).map((c) => [c.n, c])), [citations]);
  const source = useMemo(() => linkifyCitations(text, new Set(byN.keys())), [text, byN]);

  const components = useMemo<Components>(
    () => ({
      a({ href, children }) {
        if (href?.startsWith(CITE_PREFIX)) {
          const c = byN.get(Number(href.slice(CITE_PREFIX.length)));
          if (c) {
            return (
              <sup>
                <button
                  type="button"
                  className="chip"
                  aria-label={`Source ${c.n}: ${c.title}`}
                  onMouseEnter={(e) => onChipEnter(c, e.currentTarget)}
                  onMouseLeave={onChipLeave}
                  onFocus={(e) => onChipEnter(c, e.currentTarget)}
                  onBlur={onChipLeave}
                  onClick={(e) => onChipClick(c, e.currentTarget)}
                >
                  {c.n}
                </button>
              </sup>
            );
          }
        }
        return (
          <a href={href} target="_blank" rel="noreferrer">
            {children}
          </a>
        );
      },
    }),
    [byN, onChipEnter, onChipLeave, onChipClick],
  );

  return (
    <div className={`answer markdown${streaming ? " streaming" : ""}`}>
      <Markdown components={components}>{source}</Markdown>
    </div>
  );
});
