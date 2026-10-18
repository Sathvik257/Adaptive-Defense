# Hosting Adaptive Defense

## What is ready

The project includes a Dockerfile, a loopback-bound Compose preview, a host entry point honoring `PORT`, pinned direct hosting dependencies, a health check, a non-root container user, and a bundled synthetic demo. The interface needs a Python server and WebSockets; it cannot run as a static GitHub Pages or Cloudflare Pages upload. No public deployment or paid resource has been created.

## Docker deployment

From the project directory with Docker Engine running:

```sh
docker compose up --build -d
docker compose ps
```

Open http://localhost:8501. Compose binds to loopback for preview. For public hosting, use a container host with HTTPS and WebSocket support, deploy this Dockerfile, and route traffic to its `PORT` (default 8501). Start with 2 vCPU and 2 GB RAM for a small demo, then measure memory and concurrency on the actual host. This allocation is a recommendation to test, not a verified minimum or throughput guarantee.

The health endpoint is `/_stcore/health`. Startup includes Python/ML imports, so the image gives a 90-second health-check grace period. No training runs at startup. The image copies only source, CSS, configuration and demo inference artifacts; local environments, private CSVs and other experiment folders are excluded. The CPU PyTorch wheel avoids installing unnecessary GPU runtime packages.

## Python server deployment

Use Python 3.14 with a platform-supported CPU PyTorch wheel:

```sh
python -m pip install -r requirements-host.txt
python serve.py
```

Set `PORT` in your hosting provider if necessary. The app binds to all interfaces inside the container; access control and HTTPS belong to the host or reverse proxy. The direct dependency pins match the saved model's local environment. Transitive dependencies are not fully locked, and availability of a matching Linux wheel must be confirmed during the first image build. Retrain artifacts if intentionally changing incompatible scikit-learn/PyTorch versions.

## Data and resource handling

- The hosted UI accepts CSVs only, limited to 10 MB, 10,000 rows and 256 columns.
- Inference results remain in the user's Streamlit session; the application does not save uploaded CSVs to disk or use a global upload cache.
- Hosting means files are transmitted to your server. Do not describe hosted inference as occurring on the visitor's computer.
- Only trusted server-owned model artifacts are loaded. Visitors cannot upload pickle or model files.
- XSRF and CORS protections remain enabled. Usage telemetry is disabled.
- Use provider authentication if the app should be private. This project does not implement accounts, rate limits, or a multi-tenant production service.
- Training remains a CLI workflow. Large experiments should be executed separately from the web process.

## Verification

```sh
python tools/check_resources.py
python -m pytest -q
```

`docs/resource-check.json` records locally measured inference and artifact size. Model-load time is included in inference measurements. Local process RSS, when available, is not a peak-memory or concurrency measurement. A successful Python test run does not prove a Docker build or remote deployment; see the verification record for actual coverage.

References checked during implementation:

- [Streamlit Docker deployment](https://docs.streamlit.io/deploy/tutorials/docker)
- [Community Cloud resource limits and dependency pinning](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

Community Cloud limits can change. Do not assume the model fits a free tier without testing. Docker is the primary prepared deployment path here.
