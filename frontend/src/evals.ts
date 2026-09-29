// Pure helpers for the Evals tab: formatting, grouping labels and the question table's
// filter and sort.

import type { EvalOutcome, EvalRecord, GroupStats, TierSummary } from "./evalTypes";

export const pct = (x: number | null | undefined): string =>
  x == null ? "–" : `${Math.round(x * 1000) / 10}%`;

export const secs = (x: number | null | undefined, digits = 1): string =>
  x == null ? "–" : `${x.toFixed(digits)}s`;

export const num = (x: number | null | undefined, digits = 0): string =>
  x == null ? "–" : x.toLocaleString(undefined, { maximumFractionDigits: digits });

/** "2026-09-29 21:38 UTC" from an ISO timestamp. */
export function when(iso: string | null | undefined): string {
  if (!iso) return "–";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return `${d.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

export const OUTCOME_ORDER: EvalOutcome[] = ["supported", "not_found", "out_of_turns", "error", "timeout"];

export const OUTCOME_SHORT: Record<EvalOutcome, string> = {
  supported: "supported",
  not_found: "not found",
  out_of_turns: "out of turns",
  error: "error",
  timeout: "timeout",
};

/** Non-zero outcomes, e.g. [["supported", 80], ["not_found", 1]]. */
export function outcomeCounts(stats: GroupStats): [EvalOutcome, number][] {
  return OUTCOME_ORDER.map((o) => [o, stats.outcomes?.[o] ?? 0] as [EvalOutcome, number]).filter(
    ([, n]) => n > 0,
  );
}

const BREAKDOWN_LABEL: Record<string, string> = {
  by_type: "By type",
  by_difficulty: "By difficulty",
  by_sequential_depth: "By sequential depth",
  by_min_turns_estimate: "By min-turns estimate",
};

/** A tier's breakdown tables in display order: [key, title, groups]. */
export function breakdowns(tier: TierSummary): [string, string, Record<string, GroupStats>][] {
  return Object.keys(tier)
    .filter((k) => k.startsWith("by_"))
    .map((k) => [
      k,
      BREAKDOWN_LABEL[k] ?? k.replace(/^by_/, "By ").replace(/_/g, " "),
      tier[k] as Record<string, GroupStats>,
    ]);
}

export type Verdict = "correct" | "partial" | "wrong";

export function verdictOf(r: EvalRecord): Verdict {
  if (r.grading?.correct) return "correct";
  return r.grading?.partially_correct ? "partial" : "wrong";
}

export type QuestionFilter = "all" | "wrong" | "unsupported";
export type QuestionSort = "id" | "time" | "turns";

export function filterAndSort(
  records: EvalRecord[],
  filter: QuestionFilter,
  sort: QuestionSort,
  query: string,
): EvalRecord[] {
  const q = query.trim().toLowerCase();
  const kept = records.filter((r) => {
    if (filter === "wrong" && verdictOf(r) === "correct") return false;
    if (filter === "unsupported" && r.outcome === "supported") return false;
    if (!q) return true;
    return [r.id, r.type, r.question, r.expected_answer, r.final_answer].some((s) =>
      s?.toLowerCase().includes(q),
    );
  });
  const key: Record<QuestionSort, (r: EvalRecord) => number | string> = {
    id: (r) => r.id,
    time: (r) => -(r.timing?.total_s ?? -1),
    turns: (r) => -(r.turns_used ?? -1),
  };
  const k = key[sort];
  return [...kept].sort((a, b) => {
    const x = k(a);
    const y = k(b);
    return x < y ? -1 : x > y ? 1 : a.id.localeCompare(b.id);
  });
}
