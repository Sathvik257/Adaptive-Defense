# React research console

The primary web application uses React 19, TypeScript, Vite, FastAPI, WebSockets
and SQLite. The original Streamlit application remains available on port 8501.

## Start locally

```powershell
python -m pip install -r requirements-api.txt
npm.cmd ci --prefix web
npm.cmd run build --prefix web
python serve_api.py
```

Open http://127.0.0.1:8000. Set HOST and PORT for your hosting platform. For frontend
development, run `npm.cmd run dev --prefix web` alongside the API; Vite proxies
HTTP and WebSocket requests. Node is needed only during the frontend build.

## Data and decisions

`tools/export_console.py` exports 530 synthetic evaluation flows from the saved
models. PCA fits training embeddings only. Details include anomaly score, RF
entropy, DQN threshold action, Q-values and reconstruction residuals. These are
not SHAP values or calibrated attack probabilities. The stream is recorded,
not a timestamped packet capture. Ground truth is omitted from API responses.

Review requests consume a twenty-query browser-session budget and persist in
SQLite. Duplicate requests do not consume another query. Labels are saved but
do not update the model. Benchmark cost controls re-score fixed predictions.

## Budget-aware research

`adaptive_defense/budget_env.py` implements Gymnasium with allow, alert and query
actions. Features, remaining budget and episode position form observations.
Labels enter offline rewards and requested feedback only. Default costs are
missed attack 10, false alarm 0.5, query 0.3. A query assumes a perfect immediate
analyst. Exhausted queries fall back to alert and pay query overhead.

`tools/train_budget_policy.py` trains SB3 PPO and evaluates frozen predictions
on a separate test set. Supply an NPZ with x_train, y_train, x_test, y_test:

```powershell
python tools/train_budget_policy.py --data data/prepared.npz --output runs/ppo --steps 10000
```

The caller must construct disjoint partitions and fit all feature transforms on
training records only. This CLI does not validate provenance or automatically
construct LOAO splits. PPO is experimental and is not the console's saved DQN.
No PPO superiority or adaptation gain is claimed.

Real UNSW-NB15 evaluation, complete leave-one-family-out sweeps, time/group
partitions, five-seed uncertainty, delayed/noisy feedback, drift, forgetting,
SHAP and live packet capture remain future research, not implemented results.

## Hosting

Build React, install Python dependencies and run `python serve_api.py` with the
platform's HOST/PORT. The API serves frontend assets from web/dist. Persist
runtime/reviews.sqlite3 (or AD_DATABASE) for review storage. Use one instance
with SQLite; multiple replicas need shared storage.

This is a demonstration service. Before public deployment add authentication,
rate limiting, HTTPS and data retention. A cookie isolates sessions but is not
an authenticated user; clearing it starts a new budget. No network enforcement
or sensitive live traffic is involved. The original Docker recipe runs Streamlit.
For React use `docker compose -f compose.web.yaml up --build`. This multi-stage
image serves the static artifact API without model training dependencies.
The review-data volume persists SQLite. Docker runtime verification is pending.

## Verification

Run `python -m pytest -q` and `npm.cmd run build --prefix web`. Tests cover budget
exhaustion, duplicate queries, session isolation, origin rejection, WebSocket
validation, cost arithmetic and Gymnasium label isolation. Motion respects
prefers-reduced-motion. Event table buttons provide keyboard flow inspection.
