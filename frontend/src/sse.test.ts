import { describe, expect, it } from "vitest";
import { SseParser, type SseMessage } from "./sse";

function parseAll(chunks: string[], end = true): SseMessage[] {
  const out: SseMessage[] = [];
  const p = new SseParser((m) => out.push(m));
  for (const c of chunks) p.feed(c);
  if (end) p.end();
  return out;
}

describe("SseParser", () => {
  it("parses named events", () => {
    const msgs = parseAll([
      'event: token\ndata: {"delta":"Hi"}\n\nevent: done\ndata: {}\n\n',
    ]);
    expect(msgs).toEqual([
      { event: "token", data: '{"delta":"Hi"}', id: undefined },
      { event: "done", data: "{}", id: undefined },
    ]);
  });

  it("handles splits at every possible chunk boundary", () => {
    const stream =
      'event: status\ndata: {"stage":"research"}\n\nevent: token\r\ndata: {"delta":"a b"}\r\n\r\n';
    for (let i = 0; i <= stream.length; i++) {
      const msgs = parseAll([stream.slice(0, i), stream.slice(i)], false);
      expect(msgs.map((m) => [m.event, m.data])).toEqual([
        ["status", '{"stage":"research"}'],
        ["token", '{"delta":"a b"}'],
      ]);
    }
  });

  it("handles one character at a time", () => {
    const stream = 'event: token\ndata: {"delta":"x"}\n\n';
    const msgs = parseAll(stream.split(""), false);
    expect(msgs).toHaveLength(1);
    expect(msgs[0].data).toBe('{"delta":"x"}');
  });

  it("joins multi-line data with newlines", () => {
    const msgs = parseAll(["event: token\ndata: line one\ndata: line two\ndata:\n\n"]);
    expect(msgs[0].data).toBe("line one\nline two\n");
  });

  it("ignores comment lines like ': ping'", () => {
    const msgs = parseAll([
      ": ping\n\n",
      "event: token\n: ping\ndata: ok\n\n",
      ":keepalive\n\n",
    ]);
    expect(msgs).toEqual([{ event: "token", data: "ok", id: undefined }]);
  });

  it("defaults the event name to 'message' and resets it between events", () => {
    const msgs = parseAll(["event: a\ndata: 1\n\ndata: 2\n\n"]);
    expect(msgs.map((m) => m.event)).toEqual(["a", "message"]);
  });

  it("only strips one leading space and keeps colons in values", () => {
    const msgs = parseAll(['data:  {"a":"b:c"}\n\n']);
    expect(msgs[0].data).toBe(' {"a":"b:c"}');
  });

  it("dispatches a trailing event without the final blank line on end()", () => {
    expect(parseAll(["event: done\ndata: {}"], false)).toHaveLength(0);
    expect(parseAll(["event: done\ndata: {}"], true)).toHaveLength(1);
  });
});
