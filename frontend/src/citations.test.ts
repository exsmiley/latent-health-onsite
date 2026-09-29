import { describe, expect, it } from "vitest";
import { linkifyCitations } from "./citations";

const known = new Set([1, 2, 3]);

describe("linkifyCitations", () => {
  it("links single, adjacent and comma-separated markers", () => {
    const out = linkifyCitations("A [1]. B [1][3]. C [2, 3].", known).replaceAll("\u2060", "^");
    expect(out).toBe("A^[1](#cite-1). B^[1](#cite-1)^[3](#cite-3). C^[2](#cite-2)^[3](#cite-3).");
  });
  it("leaves unknown markers and code alone", () => {
    expect(linkifyCitations("x [9] `a[1]`", known)).toBe("x [9] `a[1]`");
  });
});
