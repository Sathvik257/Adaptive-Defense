# Deploy the original console on Vercel

**Live demo:** [web-flame-eight-75.vercel.app](https://web-flame-eight-75.vercel.app/)

This is a static recorded-synthetic demo. It runs replay, inspection, benchmarks, cost comparisons and a 20-item analyst queue entirely in the browser. No Python runtime, external database, credentials or model weights are needed. It does not train models, capture traffic or perform live inference. Reviews use sessionStorage: they survive reloads in the same tab but are not server-backed, shared across devices, or a tamper-proof budget. Closing the tab clears its session. Browser storage must be enabled.

## GitHub import

1. Import `Sathvik257/Adaptive-Defense` into Vercel.
2. Keep Root Directory at the repository root (not `web`).
3. Use the **Services** preset. The committed vercel.json explicitly declares one Vite service named `web`, rooted at `.` so it can read the recorded artifact. Its install command is `npm ci --prefix web`, build is `npm run build:vercel --prefix web`, and output is `web/dist`. The catch-all public rewrite targets `web`. There is no Python service and no internal binding: hosted mode makes no backend calls.
4. Use Node.js 22 or 24. No environment variables are required.
5. Refresh or restart the GitHub import after the configuration commit reaches main. If the import still lists a Python service named `app`, it is showing the old auto-detected setup; re-import the latest main revision before deploying. Deploy and verify Start/Pause/Reset, review labels and Benchmarks.

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

The user deployed the live demo linked above. Its homepage returned HTTP 200 when checked on 2 November 2026. Local build checks remain distinct from Vercel deployment logs; consult the Vercel dashboard for build status and configuration.

Reference: https://vercel.com/docs/frameworks/frontend/vite
