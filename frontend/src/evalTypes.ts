// Eval-run data served by GET /api/evals and GET /api/evals/{id}: the files `rag eval` writes
// (evals/README.md, "Output"). Fields are optional where older or unfinished runs may lack them.

export type EvalOutcome = "supported" | "not_found" | "out_of_turns" | "error" | "timeout";

export interface EvalConfig {
  label?: string | null;
  set?: string | null;
  file?: string | null;
  n_questions?: number;
  max_turns?: number;
  concurrency?: number;
  judge?: boolean;
  chat_model?: string;
  git_commit?: string | null;
  started_at?: string | null;
  finished_at?: string;
  wall_clock_s?: number;
}

/** report.group_stats(): stats for one group of questions. */
export interface GroupStats {
  n: number;
  accuracy: number | null;
  exact_rate: number | null;
  partial_rate: number | null;
  outcomes: Record<EvalOutcome, number>;
  turns_mean: number | null;
  turns_p90: number | null;
  evaluator_rejections_mean: number | null;
  time_s: { median: number | null; p90: number | null; max: number | null; mean: number | null };
  first_token_s_median: number | null;
  research_turn_s_mean: number | null;
  research_model_s_mean: number | null;
  research_tools_s_mean: number | null;
  evaluator_call_s_mean: number | null;
  responder_first_token_s_median: number | null;
  responder_total_s_median: number | null;
  tokens_mean: Record<string, number | null>;
  cost_usd_mean: number | null;
  cost_usd_total: number | null;
  citation_recall_mean: number | null;
  citation_precision_mean: number | null;
  citation_any_overlap_rate: number | null;
}

/** Per tier: `all`, `by_type`, and when present `by_difficulty`, `by_sequential_depth`, ... */
export type TierSummary = { all: GroupStats } & Record<string, GroupStats | Record<string, GroupStats>>;

export interface EvalSummary {
  overall: GroupStats;
  by_tier: Record<string, TierSummary>;
}

export interface EvalRunListItem {
  id: string;
  complete: boolean;
  config: EvalConfig;
  overall: GroupStats | null;
}

export interface EvalRecord {
  id: string;
  tier?: string;
  type?: string;
  difficulty?: string;
  hops?: number;
  sequential_depth?: number | null;
  question?: string;
  expected_answer?: string;
  answer_aliases?: string[];
  outcome: EvalOutcome;
  turns_used?: number | null;
  error?: string | null;
  final_answer?: string | null;
  evaluations?: { turn: number; verdict: string; feedback?: string }[];
  tool_calls?: { turn: number; name: string }[];
  evaluator_rejections?: number;
  timing?: {
    total_s?: number | null;
    first_token_s?: number | null;
    research_turns?: { turn: number; s: number; model_s?: number | null; tools_s?: number | null }[];
    evaluator_calls?: { turn: number; s: number }[];
    responder_first_token_s?: number | null;
    responder_total_s?: number | null;
  };
  usage?: { input_tokens?: number; cached_input_tokens?: number; output_tokens?: number };
  citations?: { n: number; chunk_id: number; title: string; section: string | null }[];
  grading?: {
    exact?: boolean;
    correct?: boolean;
    partially_correct?: boolean;
    method?: string;
    judge?: { reason?: string; error?: string } | null;
    citation_recall?: number | null;
  };
}

export interface EvalRun {
  id: string;
  complete: boolean;
  config: EvalConfig;
  summary: EvalSummary;
  records: EvalRecord[];
}
