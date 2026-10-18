import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Shield,
  Activity,
  ScanLine,
  Inbox,
  BarChart3,
  Play,
  Pause,
  RotateCcw,
  ArrowUpRight,
  CircleDot,
} from "lucide-react";
import "./style.css";
import { TechniqueReference } from "./TechniqueReference";
import { ResearchEvidence } from "./ResearchEvidence";

type Flow = {
  id: string;
  index: number;
  score: number;
  entropy: number;
  x: number;
  y: number;
  action: string;
  threshold: number;
  threshold_action: string;
  q_values: number[];
  contributions: { feature: string; error: number }[];
};
type Metric = {
  model: string;
  recall: number;
  f1: number;
  false_positive_rate: number;
  unseen_recall: number;
};
type Experiment = {
  dataset: string;
  projection: string;
  policy: string;
  flows: Flow[];
  metrics: Metric[];
};
type Queue = {
  budget: number;
  used: number;
  remaining: number;
  reviews: { flow_id: string; label: string | null }[];
};
async function api<T>(
  path: string,
  method = "GET",
  body?: unknown,
): Promise<T> {
  const res = await fetch("/api/" + path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let message = await res.text();
    try {
      message = JSON.parse(message).detail;
    } catch {}
    throw Error(message);
  }
  return res.json();
}
const pct = (n: number) => (n * 100).toFixed(1) + "%";
function App() {
  const [data, setData] = useState<Experiment | null>(null),
    [error, setError] = useState(""),
    [tab, setTab] = useState("Operations"),
    [count, setCount] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(180),
    [selected, setSelected] = useState<Flow | null>(null),
    [queue, setQueue] = useState<Queue | null>(null),
    [cost, setCost] = useState(10),
    [costs, setCosts] = useState<{ model: string; cost: number }[]>([]);
  const cursor = useRef(0);
  const refresh = () => api<Queue>("reviews").then(setQueue);
  useEffect(() => {
    Promise.all([
      api<Experiment>("experiment").then(setData),
      api("session", "POST").then(refresh),
    ]).catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    if (!playing || !data) return;
    let timer: number | undefined;
    let closed = false;
    const ws = new WebSocket(
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/replay`,
    );
    const pull = () => {
      if (!closed && ws.readyState === WebSocket.OPEN)
        ws.send(JSON.stringify({ index: cursor.current }));
    };
    ws.onopen = pull;
    ws.onmessage = (event) => {
      if (closed) return;
      const packet = JSON.parse(event.data);
      if (packet.error) {
        setError(packet.error);
        setPlaying(false);
        return;
      }
      cursor.current = packet.flow.index + 1;
      setCount(cursor.current);
      setSelected(packet.flow);
      if (cursor.current >= data.flows.length) setPlaying(false);
      else timer = window.setTimeout(pull, speed);
    };
    ws.onerror = () => {
      setError("Replay connection lost. Press play to reconnect.");
      setPlaying(false);
    };
    ws.onclose = () => {
      if (!closed) {
        setPlaying(false);
      }
    };
    return () => {
      closed = true;
      clearTimeout(timer);
      ws.close();
    };
  }, [playing, speed, data]);
  useEffect(() => {
    let active = true;
    api<{ results: { model: string; cost: number }[] }>("costs", "POST", {
      missed_attack: cost,
      false_alarm: 0.5,
      query: 0.3,
    })
      .then((r) => {
        if (active) setCosts(r.results);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [cost]);
  async function ask(flow: Flow) {
    try {
      await api("reviews/" + flow.id, "POST");
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function label(id: string, value: string) {
    try {
      await api("reviews/" + id, "PUT", { label: value });
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const visible = data?.flows.slice(0, count) || [],
    alerts = visible.filter((f) => f.action === "alert").length;
  const extent = (key: "x" | "y") => {
    const values = data?.flows.map((f) => f[key]) || [0, 1];
    return [Math.min(...values), Math.max(...values)];
  };
  const [xmin, xmax] = extent("x"),
    [ymin, ymax] = extent("y");
  return (
    <div className="shell">
      <aside>
        <a className="brand" href="/">
          <Shield size={27} />
          <span>
            adaptive<span className="brand-sub">DEFENSE / RESEARCH</span>
          </span>
        </a>
        <div className="nav-label">WORKSPACE</div>
        <nav>
          {[
            [Activity, "Operations"],
            [Inbox, "Analyst queue"],
            [BarChart3, "Benchmarks"],
          ].map(([Icon, name]) => {
            const I = Icon as typeof Activity;
            return (
              <button
                key={String(name)}
                className={tab === name ? "active" : ""}
                onClick={() => setTab(String(name))}
              >
                <I size={18} />
                {String(name)}
                {name === "Analyst queue" && <small>{queue?.used || 0}</small>}
              </button>
            );
          })}
        </nav>
        <div className="side-note">
          <CircleDot size={18} />
          <strong>Research environment</strong>
          <p>
            Recorded synthetic traffic.
            <br />
            No live network enforcement.
          </p>
          <span>BUILD 03 / LOCAL CONSOLE</span>
        </div>
      </aside>
      <main>
        <header>
          <span>
            Workspace <b>/ {tab}</b>
          </span>
          <span className="connection">
            <i />
            {data ? "Artifact connected" : "Connecting"}
          </span>
        </header>
        <section className="page">
          <div className="heading">
            <div className="eyebrow">ADAPTIVE DEFENSE LAB</div>
            <h1>
              {tab === "Operations"
                ? "Observe. Understand. Respond."
                : tab === "Analyst queue"
                  ? "Human judgment, within budget."
                  : "Measure the trade-offs."}
            </h1>
            <p>
              {tab === "Operations"
                ? "Follow each recorded decision through the detection pipeline."
                : tab === "Analyst queue"
                  ? "Review uncertain flows and record feedback for future training."
                  : "Compare the saved experiment before drawing conclusions."}
            </p>
          </div>
          {error && (
            <div className="error" role="alert">
              {error}
              <button onClick={() => setError("")}>Dismiss</button>
            </div>
          )}
          {!data ? (
            <div className="panel empty">Loading experiment…</div>
          ) : (
            <>
              <div className="stats">
                <div>
                  <span>PROCESSED FLOWS</span>
                  <strong>
                    {count.toString().padStart(3, "0")}
                    <small> / {data.flows.length}</small>
                  </strong>
                  <p>Recorded evaluation stream</p>
                </div>
                <div>
                  <span>DETECTION ALERTS</span>
                  <strong>{alerts.toString().padStart(3, "0")}</strong>
                  <p>Frozen DQN decisions</p>
                </div>
                <div>
                  <span>ANALYST BUDGET</span>
                  <strong>
                    {queue?.remaining ?? 20}
                    <small> queries left</small>
                  </strong>
                  <p>{queue?.used || 0} of 20 allocated</p>
                </div>
                <div>
                  <span>DATA PROVENANCE</span>
                  <strong className="text-stat">
                    Synthetic<span className="tag">DEMO</span>
                  </strong>
                  <p>Real model outputs · simulated data</p>
                </div>
              </div>
              {tab === "Operations" && (
                <>
                  <div className="toolbar">
                    <span>
                      <i className={playing ? "pulse" : ""} />
                      {playing
                        ? "Replay running"
                        : count === data.flows.length
                          ? "Replay complete"
                          : "Replay paused"}
                    </span>
                    <div>
                      <label>
                        Speed{" "}
                        <select
                          aria-label="Replay speed"
                          value={speed}
                          onChange={(e) => setSpeed(Number(e.target.value))}
                        >
                          <option value={600}>Slow</option>
                          <option value={180}>Normal</option>
                          <option value={40}>Fast</option>
                        </select>
                      </label>
                      <button
                        aria-label="Reset replay"
                        onClick={() => {
                          setPlaying(false);
                          cursor.current = 0;
                          setCount(0);
                          setSelected(null);
                        }}
                      >
                        <RotateCcw size={16} />
                      </button>
                      <button
                        className="primary"
                        disabled={count === data.flows.length}
                        onClick={() => setPlaying(!playing)}
                      >
                        {playing ? <Pause size={16} /> : <Play size={16} />}{" "}
                        {playing ? "Pause" : "Start replay"}
                      </button>
                    </div>
                  </div>
                  <div className="workspace">
                    <section className="panel radar">
                      <div className="panel-title">
                        <h2>
                          <ScanLine size={18} />
                          Traffic landscape
                        </h2>
                        <span>TRAIN-FITTED PCA</span>
                      </div>
                      <div className="chart">
                        <svg
                          viewBox="0 0 700 350"
                          role="img"
                          aria-label="Autoencoder embedding of replayed flows"
                        >
                          <defs>
                            <pattern
                              id="grid"
                              width="35"
                              height="35"
                              patternUnits="userSpaceOnUse"
                            >
                              <path
                                d="M35 0H0V35"
                                fill="none"
                                stroke="#26332e"
                                strokeWidth=".6"
                              />
                            </pattern>
                          </defs>
                          <rect width="700" height="350" fill="url(#grid)" />
                          {visible.map((f) => (
                            <circle
                              key={f.id}
                              cx={
                                30 + ((f.x - xmin) / (xmax - xmin || 1)) * 640
                              }
                              cy={
                                320 - ((f.y - ymin) / (ymax - ymin || 1)) * 290
                              }
                              r={selected?.id === f.id ? 6 : 3.5}
                              fill={
                                f.action === "alert" ? "#e6b46c" : "#83c6ad"
                              }
                              opacity={selected?.id === f.id ? 1 : 0.65}
                              onClick={() => setSelected(f)}
                            >
                              <title>
                                {f.id}: {f.action}
                              </title>
                            </circle>
                          ))}
                        </svg>
                        {count === 0 && (
                          <div className="chart-empty">
                            <ScanLine size={30} />
                            <strong>Your traffic story starts here</strong>
                            <span>
                              Start replay to reveal the recorded flows.
                            </span>
                          </div>
                        )}
                      </div>
                      <div className="legend">
                        <span>
                          <i />
                          Allowed
                        </span>
                        <span>
                          <i className="amber" />
                          Alerted
                        </span>
                        <span>
                          Proximity shows embedding similarity, not proof of
                          attack.
                        </span>
                      </div>
                    </section>
                    <section className="panel inspector">
                      <div className="panel-title">
                        <h2>Decision detail</h2>
                        <ArrowUpRight size={17} />
                      </div>
                      {selected ? (
                        <>
                          <div className="flow-title">
                            <code>{selected.id}</code>
                            <span className={"badge " + selected.action}>
                              {selected.action}
                            </span>
                          </div>
                          <dl>
                            <div>
                              <dt>Anomaly score</dt>
                              <dd>{selected.score.toFixed(4)}</dd>
                            </div>
                            <div>
                              <dt>Active threshold</dt>
                              <dd>{selected.threshold.toFixed(4)}</dd>
                            </div>
                            <div>
                              <dt>Classifier entropy</dt>
                              <dd>{selected.entropy.toFixed(3)} bits</dd>
                            </div>
                          </dl>
                          <h3>
                            Threshold action · {selected.threshold_action}
                          </h3>
                          <div className="qvalues">
                            {selected.q_values.map((q, i) => (
                              <div key={i}>
                                <span>{["Lower", "Hold", "Raise"][i]}</span>
                                <strong>{q.toFixed(2)}</strong>
                              </div>
                            ))}
                          </div>
                          <p className="fine">
                            Q-values estimate return, not probabilities.
                          </p>
                          <h3>Largest reconstruction errors</h3>
                          {selected.contributions.map((c) => (
                            <div className="contribution" key={c.feature}>
                              <span>{c.feature}</span>
                              <code>{c.error.toFixed(3)}</code>
                            </div>
                          ))}
                          <p className="fine">
                            Residuals explain reconstruction error; these are
                            not SHAP attributions.
                          </p>
                          <button
                            className="wide"
                            disabled={
                              !queue ||
                              queue.remaining === 0 ||
                              queue.reviews.some(
                                (r) => r.flow_id === selected.id,
                              )
                            }
                            onClick={() => ask(selected)}
                          >
                            <Inbox size={16} />
                            Request analyst review
                          </button>
                        </>
                      ) : (
                        <div className="empty">
                          Select a flow after starting replay to inspect its
                          decision.
                        </div>
                      )}
                    </section>
                  </div>
                  <section className="panel feed">
                    <div className="panel-title">
                      <h2>Event stream</h2>
                      <span>LATEST 12 RECORDS</span>
                    </div>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Flow</th>
                            <th>Anomaly score</th>
                            <th>Threshold action</th>
                            <th>Decision</th>
                            <th>Inspect</th>
                          </tr>
                        </thead>
                        <tbody>
                          {visible
                            .slice(-12)
                            .reverse()
                            .map((f) => (
                              <tr
                                key={f.id}
                                className={
                                  selected?.id === f.id ? "selected" : ""
                                }
                              >
                                <td>
                                  <code>{f.id}</code>
                                </td>
                                <td>{f.score.toFixed(4)}</td>
                                <td>{f.threshold_action}</td>
                                <td>
                                  <span className={"badge " + f.action}>
                                    {f.action}
                                  </span>
                                </td>
                                <td>
                                  <button
                                    aria-label={"Inspect " + f.id}
                                    onClick={() => setSelected(f)}
                                  >
                                    <ArrowUpRight size={16} />
                                  </button>
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                      {!count && (
                        <p className="empty">
                          Waiting for replay. No events have been processed.
                        </p>
                      )}
                    </div>
                  </section>
                </>
              )}
              {tab === "Analyst queue" && (
                <section className="panel">
                  <div className="panel-title">
                    <h2>Review inbox</h2>
                    <span>{queue?.remaining} QUERIES REMAINING</span>
                  </div>
                  <p className="notice">
                    Feedback is saved to this browser session’s SQLite queue. It
                    does not update the deployed model. A new-family label is an
                    analyst hypothesis.
                  </p>
                  {!queue?.reviews.length ? (
                    <div className="empty">
                      Your queue is empty. Request a review from a flow’s
                      decision detail.
                    </div>
                  ) : (
                    queue.reviews.map((r) => (
                      <div className="review" key={r.flow_id}>
                        <div>
                          <code>{r.flow_id}</code>
                          <p>
                            {r.label ? "Feedback saved" : "Awaiting assessment"}
                          </p>
                        </div>
                        <label>
                          Assessment{" "}
                          <select
                            value={r.label || ""}
                            onChange={(e) => label(r.flow_id, e.target.value)}
                          >
                            <option value="" disabled>
                              Select label
                            </option>
                            <option value="benign">Benign</option>
                            <option value="attack">Attack</option>
                            <option value="new-family">
                              Possible new family
                            </option>
                          </select>
                        </label>
                      </div>
                    ))
                  )}
                </section>
              )}
              {tab === "Benchmarks" && <TechniqueReference />}
              {tab === "Benchmarks" && <ResearchEvidence />}
              {tab === "Benchmarks" && (
                <section className="panel">
                  <div className="panel-title">
                    <h2>Saved evaluation</h2>
                    <span>SYNTHETIC / SINGLE SEED</span>
                  </div>
                  <p className="notice">
                    Attack families were withheld from initial training. The
                    adapted DQN has seen separate adaptation labels. These
                    results do not establish real-world zero-day detection or
                    PPO performance.
                  </p>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Model</th>
                          <th>Recall</th>
                          <th>False positives</th>
                          <th>Unseen recall</th>
                          <th>F1</th>
                          <th>Cost</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.metrics.map((m) => (
                          <tr key={m.model}>
                            <td>{m.model}</td>
                            <td>{pct(m.recall)}</td>
                            <td>{pct(m.false_positive_rate)}</td>
                            <td>{pct(m.unseen_recall)}</td>
                            <td>{pct(m.f1)}</td>
                            <td>
                              {costs
                                .find((c) => c.model === m.model)
                                ?.cost.toFixed(1) ?? "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="cost-control">
                    <label>
                      Missed attack cost <strong>{cost}</strong>
                      <input
                        aria-label="Missed attack cost"
                        type="range"
                        min="1"
                        max="30"
                        value={cost}
                        onChange={(e) => setCost(Number(e.target.value))}
                      />
                    </label>
                    <p>
                      Cost = missed attacks × {cost} + false alarms × 0.5.
                      Re-scores fixed decisions; does not retrain. Analyst
                      review costs are excluded from this comparison.
                    </p>
                  </div>
                </section>
              )}
              <footer>
                <span>
                  <Shield size={14} />
                  Adaptive Defense · Research console
                </span>
                <span>Evidence before claims. Human review before action.</span>
              </footer>
            </>
          )}
        </section>
      </main>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
