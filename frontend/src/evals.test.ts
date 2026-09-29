import { describe, expect, it } from "vitest";
import { breakdowns, filterAndSort, pct, secs, when } from "./evals";
import type { EvalRecord, GroupStats, TierSummary } from "./evalTypes";

const rec = (id: string, over: Partial<EvalRecord> = {}): EvalRecord => ({
  id,
  outcome: "supported",
  turns_used: 2,
  timing: { total_s: 5 },
  grading: { correct: true },
  ...over,
});

describe("filterAndSort", () => {
  const records = [
    rec("q2", { timing: { total_s: 9 }, turns_used: 4 }),
    rec("q1", {
      grading: { correct: false, partially_correct: true },
      question: "Who built the Eiffel Tower?",
    }),
    rec("q3", { outcome: "not_found", grading: { correct: false } }),
  ];
  const ids = (rs: EvalRecord[]) => rs.map((r) => r.id);

  it("sorts by id, slowest, or most turns", () => {
    expect(ids(filterAndSort(records, "all", "id", ""))).toEqual(["q1", "q2", "q3"]);
    expect(ids(filterAndSort(records, "all", "time", ""))).toEqual(["q2", "q1", "q3"]);
    expect(ids(filterAndSort(records, "all", "turns", ""))).toEqual(["q2", "q1", "q3"]);
  });

  it("filters wrong (incl. partial) and unsupported answers, and by text", () => {
    expect(ids(filterAndSort(records, "wrong", "id", ""))).toEqual(["q1", "q3"]);
    expect(ids(filterAndSort(records, "unsupported", "id", ""))).toEqual(["q3"]);
    expect(ids(filterAndSort(records, "all", "id", "eiffel"))).toEqual(["q1"]);
  });
});

describe("formatting", () => {
  it("formats rates, seconds and times", () => {
    expect(pct(0.988)).toBe("98.8%");
    expect(pct(null)).toBe("–");
    expect(secs(10.34)).toBe("10.3s");
    expect(when("2026-09-29T21:38:02+00:00")).toBe("2026-09-29 21:38 UTC");
  });

  it("lists a tier's breakdowns with readable titles", () => {
    const g = {} as GroupStats;
    const tier = { all: g, by_type: { a: g }, by_sequential_depth: { "6": g } } as unknown as TierSummary;
    expect(breakdowns(tier).map(([k, title]) => [k, title])).toEqual([
      ["by_type", "By type"],
      ["by_sequential_depth", "By sequential depth"],
    ]);
  });
});
