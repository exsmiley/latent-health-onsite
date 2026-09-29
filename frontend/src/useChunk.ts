import { useEffect, useState } from "react";
import { fetchChunk } from "./api";
import type { Chunk } from "./types";

export type ChunkState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ok"; chunk: Chunk }
  | { status: "error"; message: string };

/** Lazily loads a chunk when `id` is non-null. Results are cached in api.fetchChunk. */
export function useChunk(id: number | null): ChunkState {
  const [state, setState] = useState<ChunkState>({ status: "idle" });
  useEffect(() => {
    if (id == null) {
      setState({ status: "idle" });
      return;
    }
    let live = true;
    setState({ status: "loading" });
    fetchChunk(id).then(
      (chunk) => live && setState({ status: "ok", chunk }),
      (e: unknown) => live && setState({ status: "error", message: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      live = false;
    };
  }, [id]);
  return state;
}
