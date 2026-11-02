# Deploy the original console on Vercel

This is a static recorded-synthetic demo. It runs replay, inspection, benchmarks, cost comparisons and a 20-item analyst queue entirely in the browser. No Python runtime, external database, credentials or model weights are needed. It does not train models, capture traffic or perform live inference. Reviews use sessionStorage: they survive reloads in the same tab but are not server-backed, shared across devices, or a tamper-proof budget. Closing the tab clears its session. Browser storage must be enabled.

## GitHub import

1. Import `Sathvik257/Adaptive-Defense` into Vercel.
2. Keep Root Directory at the repository root (not `web`).
3. Select Framework Preset **Other**. The committed vercel.json sets install to `npm ci --prefix web`, build to `npm run build:vercel --prefix web`, and output to `web/dist`.
4. Use Node.js 22 or 24. No environment variables are required.
5. Deploy and verify Start/Pause/Reset, review labels and Benchmarks.

Only web/dist is published. The hosted build emits a whitelist of console artifact fields, excluding the top-level ground-truth records. Do not add private data to this synthetic artifact. No model weights or Python packages enter the static output.

## Local verification

```powershell
npm ci --prefix web
npm run build:vercel --prefix web
npm run preview --prefix web -- --port 4173
```

Open http://127.0.0.1:4173 with the API server stopped: no /api or WebSocket requests should occur. Browser storage permissions may cause an explicit error if reviews cannot be saved.

## Existing local backend mode

`npm run build --prefix web` still builds the original FastAPI/WebSocket/SQLite client. Run `python serve_api.py` as before. The hosted adapter is selected at build time only; no credentials go in VITE variables. The alternate light-theme review branch is not merged.

Hosting configuration and local browser checks do not prove a successful public Vercel deployment. No site was published as part of preparing this configuration.

Reference: https://vercel.com/docs/frameworks/frontend/vite
