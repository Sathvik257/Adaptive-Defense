# Architecture and project map

Adaptive Defense uses Python, PyTorch, scikit-learn, pandas, Streamlit and CSS. This stack keeps experiments and inference in one language and avoids an unnecessary JavaScript API layer for the current offline workflow. Docker packages the server. GitHub Actions runs checks on pushes and pull requests.

## Dependency direction

`dashboard.py → ui/app.py → focused views → inference / experiment artifacts`

Training code does not import UI code. The experiment reader does not depend on Streamlit. Uploaded data passes through bounded parsing before inference. The application renders only the selected view, reducing unnecessary work and stopping replay when leaving its screen.

## Files

| Location | Responsibility |
|---|---|
| `dashboard.py` | Minimal Streamlit entry point |
| `adaptive_defense/ui/app.py` | Navigation, experiment selection and shared layout |
| `adaptive_defense/ui/overview.py` | Metrics, fixed-baseline deltas and outcome explanation |
| `adaptive_defense/ui/analysis.py` | Comparison plot, confusion counts and decision explorer |
| `adaptive_defense/ui/replay.py` | Play/pause/reset for saved traffic windows |
| `adaptive_defense/ui/upload.py` | CSV analysis and session-scoped result download |
| `adaptive_defense/ui/method.py` | Methodology and configuration |
| `adaptive_defense/run_store.py` | Experiment discovery and artifact validation |
| `adaptive_defense/data.py` | Schema, train-only preprocessing and category holdouts |
| `adaptive_defense/models.py` | Autoencoder and DQN learning |
| `adaptive_defense/environment.py` | Sequential threshold-control environment |
| `adaptive_defense/experiment.py` | Training, comparisons and saved artifacts |
| `adaptive_defense/predict.py` | Inference without training optimizers |
| `assets/style.css` | Color, typography, layout and accessible motion |
| `tests/` | Data boundaries, inference, upload, artifact and UI checks |
| `.github/workflows/ci.yml` | Remote test workflow |

## Design decisions

- Warm neutral surfaces and muted green keep attention on results.
- The compact `ad.` mark provides identity without decorative imagery.
- Metrics compare with a named baseline, including correct lower-is-better treatment for false alarms.
- The decision explorer exposes misses and false alarms instead of hiding them in averages.
- Reduced-motion CSS respects accessibility preferences; replay is explicitly started by the user.
- There is no live capture, firewall action or verified real zero-day claim.

## Development

Use the repository checkout for the dashboard because its CSS and run artifacts live beside the Python package. `pip install -e ".[dev]"` enables editable installation and training/inference command aliases. For the exact demonstration artifact environment use `requirements-host.txt`; see HOSTING.md for version and Linux-build limits.

To extend a view, edit its module and run UI tests. To alter model behavior, update inference/training tests and collect fresh results rather than modifying saved metric files manually.
