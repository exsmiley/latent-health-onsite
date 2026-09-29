// "Dev mode" shows the research trace with its timings; off, a message shows just "Thinking…"
// until the answer streams. On by default; the choice is remembered per browser.

import { createContext, useContext, useEffect, useState } from "react";

const KEY = "devMode";

export const DevModeContext = createContext(true);

export const useDevMode = () => useContext(DevModeContext);

/** The dev-mode setting and its setter, persisted in localStorage when available. */
export function useDevModeSetting(): [boolean, (on: boolean) => void] {
  const [on, setOn] = useState(() => {
    try {
      return localStorage.getItem(KEY) !== "0";
    } catch {
      return true;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(KEY, on ? "1" : "0");
    } catch {
      // Storage unavailable (private mode, blocked): the setting just isn't remembered.
    }
  }, [on]);
  return [on, setOn];
}
