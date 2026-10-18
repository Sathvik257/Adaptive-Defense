import report from "./research-results.json";

const percent = (value: number) => (100 * value).toFixed(1) + "%";
export function ResearchEvidence() {
  const maxDistance =
    Math.max(...report.points.map((point) => point.distance_mean), 1) * 1.1;
  const colors = ["#83c6ad", "#e6b46c", "#a9a2d9"];
  return (
    <section className="panel" style={{ marginTop: 20 }}>
      <div className="panel-title">
        <h2>Unfamiliar-family detectability</h2>
        <span>{report.folds} RUNS · SYNTHETIC</span>
      </div>
      <p className="notice">
        Five seeds per held-out family. Each fold retrains the representation
        and controller. Distance uses known training attack centroids only; this
        is an exploratory association, not proof of zero-day protection.
      </p>
      <svg
        viewBox="0 0 700 300"
        role="img"
        aria-label="Mean latent distance versus held-out recall, with one point per family and seed"
        style={{ width: "100%", maxHeight: 340 }}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((value) => (
          <g key={value}>
            <line
              x1="65"
              x2="670"
              y1={245 - 210 * value}
              y2={245 - 210 * value}
              stroke="#304135"
            />
            <text x="18" y={249 - 210 * value} fill="#9cad9f" fontSize="11">
              {percent(value)}
            </text>
          </g>
        ))}
        {report.points.map((point) => (
          <circle
            key={point.family + point.seed}
            cx={65 + (point.distance_mean / maxDistance) * 590}
            cy={245 - 210 * point.heldout_recall}
            r="5"
            opacity=".8"
            fill={
              colors[
                report.families.findIndex((f) => f.family === point.family)
              ]
            }
          >
            <title>
              {point.family}, seed {point.seed}: distance{" "}
              {point.distance_mean.toFixed(2)}, recall{" "}
              {percent(point.heldout_recall)}
            </title>
          </circle>
        ))}
        <text x="65" y="268" fill="#9cad9f" fontSize="11">
          0
        </text>
        <text x="640" y="268" fill="#9cad9f" fontSize="11">
          {maxDistance.toFixed(1)}
        </text>
        <text x="180" y="292" fill="#9cad9f" fontSize="11">
          Mean distance to nearest known attack centroid
        </text>
      </svg>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Held-out family</th>
              <th>Frozen DQN recall ± SD</th>
              <th>Random Forest recall</th>
              <th>Mean distance</th>
            </tr>
          </thead>
          <tbody>
            {report.families.map((family, index) => (
              <tr key={family.family}>
                <td>
                  <span style={{ color: colors[index] }}>● </span>
                  {family.family}
                </td>
                <td>
                  {percent(family.recall_mean)} ± {percent(family.recall_std)}
                </td>
                <td>{percent(family.rf_recall_mean)}</td>
                <td>{family.distance_mean.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="notice">
        Sample SD across seeds is not a confidence interval. Each fold has a
        different encoder, so standardized distances are not guaranteed to be
        directly comparable.
      </p>
      <div className="panel-title">
        <h2>Measured ONNX CPU latency</h2>
        <span>FLOAT32 · {report.latency.threads} THREAD</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Batch size</th>
              <th>Median batch latency</th>
              <th>p95 batch latency</th>
              <th>Flows / second</th>
            </tr>
          </thead>
          <tbody>
            {report.latency.results.map((row) => (
              <tr key={row.batch_size}>
                <td>{row.batch_size}</td>
                <td>{row.median_batch_ms.toFixed(3)} ms</td>
                <td>{row.p95_batch_ms.toFixed(3)} ms</td>
                <td>{Math.round(row.flows_per_second).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="notice">
        {report.latency.hardware}. {report.latency.scope}. This is a local CPU
        result, not a Raspberry Pi or end-to-end NIDS measurement.
      </p>
    </section>
  );
}
