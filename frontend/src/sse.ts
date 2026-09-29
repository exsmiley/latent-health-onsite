// Minimal, robust Server-Sent Events parser (per the WHATWG spec's line rules).
// Feed it arbitrary text chunks; it emits complete events as they are dispatched.

export interface SseMessage {
  /** Event name (defaults to "message" when no `event:` field was given). */
  event: string;
  /** Data lines joined with "\n". */
  data: string;
  id?: string;
}

export class SseParser {
  private buffer = "";
  private eventName = "";
  private dataLines: string[] = [];
  private lastId: string | undefined;
  private sawData = false;
  private readonly onMessage: (msg: SseMessage) => void;

  constructor(onMessage: (msg: SseMessage) => void) {
    this.onMessage = onMessage;
  }

  /** Push a decoded text chunk. Lines may be split anywhere across calls. */
  feed(chunk: string): void {
    this.buffer += chunk;
    // Process every complete line. A line ends at \n, \r\n or \r.
    let start = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      const ch = this.buffer[i];
      if (ch === "\n" || ch === "\r") {
        // A lone \r at the very end might be followed by \n in the next chunk.
        if (ch === "\r" && i === this.buffer.length - 1) break;
        const line = this.buffer.slice(start, i);
        if (ch === "\r" && this.buffer[i + 1] === "\n") i++;
        start = i + 1;
        this.processLine(line);
      }
    }
    this.buffer = this.buffer.slice(start);
  }

  /** Call at end of stream. Dispatches a trailing event lacking the final blank line. */
  end(): void {
    const rest = this.buffer.endsWith("\r") ? this.buffer.slice(0, -1) : this.buffer;
    this.buffer = "";
    if (rest.length > 0) this.processLine(rest);
    this.dispatch();
  }

  private processLine(line: string): void {
    if (line === "") {
      this.dispatch();
      return;
    }
    if (line.startsWith(":")) return; // comment, e.g. ": ping"

    const colon = line.indexOf(":");
    let field: string;
    let value: string;
    if (colon === -1) {
      field = line;
      value = "";
    } else {
      field = line.slice(0, colon);
      value = line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
    }

    switch (field) {
      case "event":
        this.eventName = value;
        break;
      case "data":
        this.dataLines.push(value);
        this.sawData = true;
        break;
      case "id":
        if (!value.includes("\0")) this.lastId = value;
        break;
      default:
        // "retry" and unknown fields are ignored.
        break;
    }
  }

  private dispatch(): void {
    if (this.sawData) {
      this.onMessage({
        event: this.eventName || "message",
        data: this.dataLines.join("\n"),
        id: this.lastId,
      });
    }
    this.eventName = "";
    this.dataLines = [];
    this.sawData = false;
  }
}
