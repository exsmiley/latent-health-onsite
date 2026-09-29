import { Fragment, useEffect, useState } from "react";
import { fetchEvalRun, fetchEvalRuns } from "../api";
import {
  breakdowns,
  filterAndSort,
  num,
  OUTCOME_SHORT,
  outcomeCounts,
  pct,
  secs,
  verdictOf,
  when,
  type QuestionFilter,
  type QuestionSort,
} from "../evals";
import type { EvalConfig, EvalRecord, EvalRun, EvalRunListItem, GroupStats } from "../evalTypes";

type Load<T> = { status: "loading" } | { status: "error"; message: string } | { status: "ok"; data: T };

/** Load `fn()` whenever `key` changes; `reload` fetches again. */
function useLoad<T>(key: string, fn: () => Promise<T>): [Load<T>, () => void] {
  const [state, setState] = useState<Load<T>>({ status: "loading" });
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    let live = true;
    setState({ status: "loading" });
    fn().then(
      (data) => live && setState({ status: "ok", data }),
      (e) => live && setState({ status: "error", message: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      live = false;
    };
    // `fn` is a fresh closure each render; `key` identifies what it loads.
  }, [key, nonce]);
  return [state, () => setNonce((n) => n + 1)];
}

/** The Evals tab: the list of runs, or one run when `runId` is set. */
export function Evals({ runId, onSelect }: { runId: string | null; onSelect: (id: string | null) => void }) {
  return (
    <main className="evals">
      {runId ? <RunDetail id={runId} onBack={() => onSelect(null)} /> : <RunList onSelect={onSelect} />}
    </main>
  );
}

const runName = (c: EvalConfig, id: string) => c.label || id;
const runSet = (c: EvalConfig) => c.set ?? (c.file ? c.file.split("/").pop() : null) ?? "–";

function RunList({ onSelect }: { onSelect: (id: string) => void }) {
  const [runs, reload] = useLoad("runs", fetchEvalRuns);

  return (
    <>
      <div className="evals-head">
        <h1>Eval runs</h1>
        <button className="ghost-btn" onClick={reload}>
          Refresh
        </button>
      </div>
      {runs.status === "loading" && <Loading />}
      {runs.status === "error" && <ErrorBox message={runs.message} />}
      {runs.status === "ok" && runs.data.length === 0 && (
        <p className="muted">
          No runs yet. Run <code>uv run rag eval --label my-run</code>; results land in{" "}
          <code>evals/results/</code>.
        </p>
      )}
      {runs.status === "ok" && runs.data.length > 0 && (
        <div className="table-wrap">
          <table className="data runs">
            <thead>
              <tr>
                <th>Run</th>
                <th>Started</th>
                <th>Set</th>
                <th className="num">Qs</th>
                <th className="num">Turns</th>
                <th className="num">Conc.</th>
                <th className="num">Accuracy</th>
                <th className="num">Exact</th>
                <th className="num">Median</th>
                <th className="num">p90</th>
                <th className="num">Avg turns</th>
                <th>Outcomes</th>
              </tr>
            </thead>
            <tbody>
              {runs.data.map((r) => (
                <RunRow key={r.id} run={r} onSelect={onSelect} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function RunRow({ run, onSelect }: { run: EvalRunListItem; onSelect: (id: string) => void }) {
  const c = run.config;
  const o = run.overall;
  return (
    <tr className="clickable" onClick={() => onSelect(run.id)}>
      <td>
        <button className="link-btn">{runName(c, run.id)}</button>
        {!run.complete && <span className="tag">unfinished</span>}
      </td>
      <td className="nowrap muted">{when(c.started_at)}</td>
      <td>{runSet(c)}</td>
      <td className="num">
        {o?.n ?? "–"}
        {c.n_questions != null && o && o.n !== c.n_questions ? ` / ${c.n_questions}` : ""}
      </td>
      <td className="num">{c.max_turns ?? "–"}</td>
      <td className="num">{c.concurrency ?? "–"}</td>
      <td className="num strong">{pct(o?.accuracy)}</td>
      <td className="num">{pct(o?.exact_rate)}</td>
      <td className="num">{secs(o?.time_s.median)}</td>
      <td className="num">{secs(o?.time_s.p90)}</td>
      <td className="num">{num(o?.turns_mean, 2)}</td>
      <td>{o && <Outcomes stats={o} />}</td>
    </tr>
  );
}

function RunDetail({ id, onBack }: { id: string; onBack: () => void }) {
  const [run, reload] = useLoad(id, () => fetchEvalRun(id));

  return (
    <>
      <div className="evals-head">
        <button className="ghost-btn" onClick={onBack}>
          ← All runs
        </button>
        <button className="ghost-btn" onClick={reload}>
          Refresh
        </button>
      </div>
      {run.status === "loading" && <Loading />}
      {run.status === "error" && <ErrorBox message={run.message} />}
      {run.status === "ok" && <RunBody run={run.data} />}
    </>
  );
}

function RunBody({ run }: { run: EvalRun }) {
  const c = run.config;
  const o = run.summary.overall;
  const tiers = Object.entries(run.summary.by_tier ?? {});
  return (
    <>
      <h1 className="run-title">
        {runName(c, run.id)}
        {!run.complete && <span className="tag">unfinished: summarized from {o.n} results so far</span>}
      </h1>
      <div className="run-meta muted">
        {[
          when(c.started_at),
          `set ${runSet(c)}`,
          c.max_turns != null && `${c.max_turns}-turn budget`,
          c.concurrency != null && `concurrency ${c.concurrency}`,
          c.chat_model,
          c.judge === false && "no judge",
          c.git_commit && `commit ${c.git_commit.slice(0, 7)}`,
          c.wall_clock_s != null && `wall clock ${secs(c.wall_clock_s)}`,
        ]
          .filter(Boolean)
          .join(" · ")}
      </div>

      <div className="tiles">
        <Tile label="Accuracy" value={pct(o.accuracy)} sub={`${pct(o.exact_rate)} exact match`} />
        <Tile
          label="Median time"
          value={secs(o.time_s.median)}
          sub={`p90 ${secs(o.time_s.p90)} · max ${secs(o.time_s.max)}`}
        />
        <Tile
          label="Turns"
          value={num(o.turns_mean, 2)}
          sub={`p90 ${num(o.turns_p90, 1)} · ${num(o.evaluator_rejections_mean, 2)} rejections`}
        />
        <Tile
          label="Tokens / question"
          value={num(o.tokens_mean?.input_tokens)}
          sub={`${num(o.tokens_mean?.cached_input_tokens)} cached · ${num(o.tokens_mean?.output_tokens)} out`}
        />
        <Tile
          label="Citations"
          value={pct(o.citation_recall_mean)}
          sub={`recall · ${pct(o.citation_precision_mean)} precision`}
        />
        {o.cost_usd_mean != null && (
          <Tile
            label="Cost / question"
            value={`$${o.cost_usd_mean.toFixed(4)}`}
            sub={`$${num(o.cost_usd_total, 2)} total`}
          />
        )}
      </div>

      <div className="run-line">
        <strong>Outcomes</strong> <Outcomes stats={o} />
      </div>
      <div className="run-line muted">
        <strong>Where the time goes</strong> research turn {secs(o.research_turn_s_mean, 2)} (model{" "}
        {secs(o.research_model_s_mean, 2)} + tools {secs(o.research_tools_s_mean, 2)}) · evaluator call{" "}
        {secs(o.evaluator_call_s_mean, 2)} · responder first token {secs(o.responder_first_token_s_median, 2)}
        , total {secs(o.responder_total_s_median, 2)}
      </div>

      {tiers.map(([tier, t]) => (
        <section key={tier} className="tier">
          {tiers.length > 1 && <h2>Tier: {tier}</h2>}
          <div className="breakdowns">
            {breakdowns(t).map(([key, title, groups]) => (
              <GroupTable key={key} title={title} groups={groups} />
            ))}
          </div>
        </section>
      ))}

      <Questions records={run.records} />
    </>
  );
}

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {sub && <div className="tile-sub muted">{sub}</div>}
    </div>
  );
}

function Outcomes({ stats }: { stats: GroupStats }) {
  return (
    <span className="outcomes">
      {outcomeCounts(stats).map(([o, n]) => (
        <span key={o} className={`pill ${o}`}>
          {n} {OUTCOME_SHORT[o]}
        </span>
      ))}
    </span>
  );
}

function GroupTable({ title, groups }: { title: string; groups: Record<string, GroupStats> }) {
  return (
    <div className="table-wrap">
      <table className="data">
        <caption>{title}</caption>
        <thead>
          <tr>
            <th />
            <th className="num">n</th>
            <th className="num">Acc.</th>
            <th className="num">Exact</th>
            <th className="num">Turns</th>
            <th className="num">Median</th>
            <th className="num">p90</th>
            <th>Outcomes</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(groups).map(([name, g]) => (
            <tr key={name}>
              <td>{name.replace(/_/g, " ")}</td>
              <td className="num">{g.n}</td>
              <td className="num strong">{pct(g.accuracy)}</td>
              <td className="num">{pct(g.exact_rate)}</td>
              <td className="num">{num(g.turns_mean, 2)}</td>
              <td className="num">{secs(g.time_s.median)}</td>
              <td className="num">{secs(g.time_s.p90)}</td>
              <td>
                <Outcomes stats={g} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const VERDICT_LABEL = { correct: "✓", partial: "½", wrong: "✗" } as const;

function Questions({ records }: { records: EvalRecord[] }) {
  const [filter, setFilter] = useState<QuestionFilter>("all");
  const [sort, setSort] = useState<QuestionSort>("id");
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const shown = filterAndSort(records, filter, sort, query);

  return (
    <section className="questions">
      <div className="questions-head">
        <h2>Questions</h2>
        <div className="controls">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as QuestionFilter)}
            aria-label="Filter"
          >
            <option value="all">All ({records.length})</option>
            <option value="wrong">Wrong or partial</option>
            <option value="unsupported">Not supported</option>
          </select>
          <select value={sort} onChange={(e) => setSort(e.target.value as QuestionSort)} aria-label="Sort">
            <option value="id">Sort by id</option>
            <option value="time">Slowest first</option>
            <option value="turns">Most turns first</option>
          </select>
          <input
            type="search"
            placeholder="Search…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search questions"
          />
        </div>
      </div>
      <div className="table-wrap">
        <table className="data questions-table">
          <thead>
            <tr>
              <th>Id</th>
              <th />
              <th>Question</th>
              <th>Type</th>
              <th>Outcome</th>
              <th className="num">Turns</th>
              <th className="num">Time</th>
              <th className="num">Tokens</th>
              <th className="num">Cit. recall</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => {
              const v = verdictOf(r);
              const isOpen = open === r.id;
              return (
                <Fragment key={r.id}>
                  <tr
                    className={`clickable${isOpen ? " open" : ""}`}
                    onClick={() => setOpen(isOpen ? null : r.id)}
                  >
                    <td className="nowrap">
                      <button className="link-btn" aria-expanded={isOpen}>
                        {r.id}
                      </button>
                    </td>
                    <td className={`verdict ${v}`} title={v}>
                      {VERDICT_LABEL[v]}
                    </td>
                    <td className="question-cell">{r.question}</td>
                    <td className="nowrap muted">{r.type?.replace(/_/g, " ")}</td>
                    <td>
                      <span className={`pill ${r.outcome}`}>{OUTCOME_SHORT[r.outcome] ?? r.outcome}</span>
                    </td>
                    <td className="num">{r.turns_used ?? "–"}</td>
                    <td className="num">{secs(r.timing?.total_s)}</td>
                    <td className="num">{num(r.usage?.input_tokens)}</td>
                    <td className="num">{pct(r.grading?.citation_recall)}</td>
                  </tr>
                  {isOpen && (
                    <tr className="detail-row">
                      <td colSpan={9}>
                        <QuestionDetail r={r} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
        {shown.length === 0 && <p className="muted">No questions match.</p>}
      </div>
    </section>
  );
}

function QuestionDetail({ r }: { r: EvalRecord }) {
  const t = r.timing ?? {};
  const judge = r.grading?.judge;
  const callsByTurn = new Map<number, string[]>();
  for (const c of r.tool_calls ?? []) callsByTurn.set(c.turn, [...(callsByTurn.get(c.turn) ?? []), c.name]);
  const verdicts = new Map((r.evaluations ?? []).map((e) => [e.turn, e.verdict]));
  const evalTime = new Map((t.evaluator_calls ?? []).map((e) => [e.turn, e.s]));

  return (
    <div className="q-detail">
      <dl>
        <dt>Expected</dt>
        <dd>
          {r.expected_answer ?? "–"}
          {r.answer_aliases && r.answer_aliases.length > 0 && (
            <span className="muted"> (also: {r.answer_aliases.join("; ")})</span>
          )}
        </dd>
        <dt>Got</dt>
        <dd>{r.final_answer || <span className="error-text">{r.error || "no answer"}</span>}</dd>
        <dt>Grading</dt>
        <dd>
          {r.grading?.method ?? "–"}
          {r.grading?.exact != null && ` · exact match ${r.grading.exact ? "yes" : "no"}`}
          {(judge?.reason || judge?.error) && <div className="muted">{judge.reason || judge.error}</div>}
        </dd>
      </dl>

      {(t.research_turns?.length ?? 0) > 0 && (
        <table className="data turns-table">
          <caption>
            Timing: total {secs(t.total_s, 2)}
            {t.first_token_s != null && ` · first token ${secs(t.first_token_s, 2)}`}
            {t.responder_total_s != null && ` · responder ${secs(t.responder_total_s, 2)}`}
          </caption>
          <thead>
            <tr>
              <th className="num">Turn</th>
              <th className="num">Research</th>
              <th className="num">Model</th>
              <th className="num">Tools</th>
              <th>Tool calls</th>
              <th>Evaluator</th>
            </tr>
          </thead>
          <tbody>
            {t.research_turns!.map((rt) => (
              <tr key={rt.turn}>
                <td className="num">{rt.turn}</td>
                <td className="num">{secs(rt.s, 2)}</td>
                <td className="num">{secs(rt.model_s, 2)}</td>
                <td className="num">{secs(rt.tools_s, 2)}</td>
                <td className="mono small">{(callsByTurn.get(rt.turn) ?? []).join(", ") || "–"}</td>
                <td>
                  {verdicts.has(rt.turn) && (
                    <>
                      <span className={`pill ${verdicts.get(rt.turn)}`}>{verdicts.get(rt.turn)}</span>{" "}
                      <span className="muted">{secs(evalTime.get(rt.turn), 2)}</span>
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {(r.citations?.length ?? 0) > 0 && (
        <div className="small">
          <strong>Cited:</strong>{" "}
          {r.citations!.map((c) => `[${c.n}] ${c.title}${c.section ? ` › ${c.section}` : ""}`).join(" · ")}
        </div>
      )}
    </div>
  );
}

function Loading() {
  return (
    <div className="trace-pending">
      <span className="spinner" /> Loading…
    </div>
  );
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="error-box" role="alert">
      <strong>Error:</strong> {message}
    </div>
  );
}
