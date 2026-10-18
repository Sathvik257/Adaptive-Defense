"""Compose the dashboard independently of training and inference."""
from pathlib import Path
import streamlit as st
from adaptive_defense.run_store import discover_runs, load_experiment
from adaptive_defense.ui import analysis, method, overview, replay, upload
from adaptive_defense.ui.components import hero

ROOT = Path(__file__).resolve().parents[2]
VIEWS = {"Overview": overview, "Compare": analysis, "Replay": replay,
         "Analyze CSV": upload, "About": method}

def main():
    st.set_page_config(page_title="Adaptive Defense", page_icon="◈", layout="wide")
    st.markdown("<style>" + (ROOT / "assets/style.css").read_text() + "</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="brand-mark">ad<span>.</span></div>', unsafe_allow_html=True)
        st.caption("ADAPTIVE DEFENSE / NETWORK LAB")
        st.divider()
        runs = discover_runs(ROOT / "runs")
        if not runs:
            st.info("No experiments yet. Run the demonstration command to get started.")
            st.code("python -m adaptive_defense.experiment --demo")
            st.stop()
        run = st.selectbox("Experiment", runs, format_func=lambda path: path.name)
        view = st.radio("Workspace", list(VIEWS), key="workspace_view")
        st.divider()
        st.caption("Explore results, compare trade-offs, and inspect individual flows.")
    try:
        experiment = load_experiment(run)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if view != "Replay":
        st.session_state["replay_play"] = False
    st.markdown('<div class="app-bar"><span>ADAPTIVE DEFENSE</span><span class="app-badge">RESEARCH EDITION</span></div>', unsafe_allow_html=True)
    hero(view, len(experiment.predictions), len(experiment.metrics))
    if experiment.is_demo:
        st.markdown('<div class="demo-note"><strong>Synthetic demo</strong> · Explore the workflow with sample traffic. Real-world detection performance has not been established.</div>', unsafe_allow_html=True)
    else:
        st.info("Offline experiment. Held-out attack categories are a proxy for unfamiliar attacks.")
    VIEWS[view].render(run, experiment.summary, experiment.metrics, experiment.predictions)
    st.markdown('<div class="footer">Adaptive Defense · Offline research · Evidence before claims</div>', unsafe_allow_html=True)
