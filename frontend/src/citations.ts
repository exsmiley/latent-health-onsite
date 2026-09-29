// Turn `[n]` / `[n, m]` markers in the answer into markdown links `#cite-n`, which the
// markdown renderer then displays as citation chips. Markers for unknown n are left as-is,
// as are code spans/blocks and markers that are already links.

export const CITE_PREFIX = "#cite-";
const WJ = "\u2060";

export function linkifyCitations(text: string, known: Set<number>): string {
  // Split out fenced code blocks and inline code so we don't touch them.
  const parts = text.split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g);
  return parts
    .map((part, i) => {
      if (i % 2 === 1) return part; // code
      return part.replace(/[ \t]*\[(\d+(?:\s*,\s*\d+)*)\](?![(:])/g, (whole, list: string) => {
        const nums = list.split(",").map((s) => Number(s.trim()));
        if (!nums.every((n) => known.has(n))) return whole;
        // U+2060 WORD JOINER keeps chips on the same line as the word they follow.
        return nums.map((n) => `${WJ}[${n}](${CITE_PREFIX}${n})`).join("");
      });
    })
    .join("");
}
