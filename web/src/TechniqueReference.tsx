/** Contextual annotations, deliberately separated from model predictions. */
export function TechniqueReference() {
  return (
    <section className="panel" style={{ marginTop: 20 }}>
      <div className="panel-title">
        <h2>ATT&CK context</h2>
        <span>REFERENCE · NOT CLASSIFICATION</span>
      </div>
      <p className="notice">
        A dataset family is not an ATT&CK technique. These analyst references
        require behavioral evidence; the binary detector does not predict
        technique IDs.
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Family / scenario</th>
              <th>Candidate technique</th>
              <th>Evidence needed</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Reconnaissance / PortScan</td>
              <td>
                <a
                  href="https://attack.mitre.org/techniques/T1046/"
                  target="_blank"
                  rel="noreferrer"
                >
                  T1046 · Network Service Discovery ↗
                </a>
              </td>
              <td>Observed internal port or service enumeration</td>
            </tr>
            <tr>
              <td>DoS</td>
              <td>
                <a
                  href="https://attack.mitre.org/techniques/T1499/"
                  target="_blank"
                  rel="noreferrer"
                >
                  T1499 · Endpoint Denial of Service ↗
                </a>
              </td>
              <td>Endpoint availability impact; network flooding differs</td>
            </tr>
            <tr>
              <td>Exploits / Fuzzers</td>
              <td>Unmapped</td>
              <td>Target, exploit behavior and adversary objective</td>
            </tr>
            <tr>
              <td>FailedLoginAttempts</td>
              <td>Unmapped</td>
              <td>Rejected attempts alone do not establish intent</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}
