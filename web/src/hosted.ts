declare const __HOSTED_DEMO__: boolean;
export const hosted = __HOSTED_DEMO__;
type Review = { flow_id: string; label: string | null };
type Artifact = {
  flows: { id: string }[];
  metrics: {
    model: string;
    fn: number;
    fp: number;
    recall: number;
    false_positive_rate: number;
  }[];
};
let artifact: Promise<Artifact> | undefined;
const key = "adaptive-defense-demo-reviews-v1";
function load(): Promise<Artifact> {
  return (artifact ??= fetch("/demo/console.json")
    .then((r) => {
      if (!r.ok) throw Error("Demo artifact unavailable. Reload to retry.");
      return r.json();
    })
    .catch((e) => {
      artifact = undefined;
      throw e;
    }));
}
function reviews(): Review[] {
  const raw = sessionStorage.getItem(key);
  if (!raw) return [];
  try {
    const value = JSON.parse(raw);
    if (
      Array.isArray(value) &&
      value.length <= 20 &&
      value.every(
        (r) =>
          typeof r.flow_id === "string" &&
          [null, "benign", "attack", "new-family"].includes(r.label),
      )
    )
      return value;
  } catch {}
  throw Error("Saved demo reviews are invalid. Open a fresh tab to restart.");
}
export async function demoApi(
  path: string,
  method: string,
  body?: unknown,
): Promise<unknown> {
  if (path === "experiment") return load();
  if (path === "session") {
    reviews();
    return { budget: 20 };
  }
  if (path === "reviews") {
    const rows = reviews();
    return {
      budget: 20,
      used: rows.length,
      remaining: 20 - rows.length,
      reviews: rows,
    };
  }
  if (path === "costs") {
    const data = await load();
    const costs = body as { missed_attack: number; false_alarm: number };
    return {
      results: data.metrics.map((m) => ({
        ...m,
        cost: m.fn * costs.missed_attack + m.fp * costs.false_alarm,
      })),
    };
  }
  if (path.startsWith("reviews/")) {
    const id = path.slice(8),
      data = await load();
    if (!data.flows.some((f) => f.id === id)) throw Error("Unknown flow");
    const rows = reviews(),
      existing = rows.find((r) => r.flow_id === id);
    if (method === "POST") {
      if (existing) return { status: "already-queued" };
      if (rows.length >= 20) throw Error("Analyst budget exhausted");
      rows.push({ flow_id: id, label: null });
    } else if (method === "PUT") {
      const label = (body as { label: string }).label;
      if (!existing) throw Error("Queue this flow before labeling it.");
      if (!["benign", "attack", "new-family"].includes(label))
        throw Error("Invalid label");
      existing.label = label;
    } else throw Error("Unsupported demo operation");
    sessionStorage.setItem(key, JSON.stringify(rows));
    return { status: "saved", model_updated: false };
  }
  throw Error("Unsupported demo operation");
}
