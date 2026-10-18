"""Small presentation components shared by workspace views."""
from html import escape
import streamlit as st


def hero(view, flow_count, model_count):
    descriptions = {
        "Overview": ("Know what gets through.", "See the signals, understand the trade-offs, and make every detection decision visible."),
        "Compare": ("Let the evidence decide.", "Compare detectors side by side, then look closely at the decisions behind the scores."),
        "Replay": ("Follow every decision.", "Move through saved traffic windows and watch the detection threshold respond."),
        "Analyze CSV": ("Put your traffic in focus.", "Upload compatible network flows and inspect the detector’s predictions."),
        "About": ("Understand the system.", "A transparent research workflow, from traffic features to an adaptive threshold."),
    }
    title, description = descriptions[view]
    st.markdown(f'''<section class="workspace-hero">
      <div class="hero-copy"><div class="eyebrow">WORKSPACE / {escape(view.upper())}</div>
      <h1>{title}</h1><p>{description}</p>
      <div class="hero-facts"><span>{flow_count:,} evaluated flows</span><span>{model_count} detectors</span><span>Offline analysis</span></div></div>
      <div class="signal-art" aria-hidden="true"><svg viewBox="0 0 250 150" focusable="false">
        <path class="signal-line" d="M15 40 H70 L110 75 H160 L200 25 H235 M15 110 H70 L110 75 M160 75 L200 125 H235" />
        <path class="signal-travel" d="M15 40 H70 L110 75 H160 L200 25 H235" />
        <circle cx="15" cy="40" r="5"/><circle cx="15" cy="110" r="5"/>
        <circle class="signal-hub" cx="125" cy="75" r="23"/><circle cx="125" cy="75" r="6"/>
        <circle cx="235" cy="25" r="5"/><circle cx="235" cy="125" r="5"/>
      </svg><span>OBSERVE / LEARN / ADAPT</span></div>
    </section>''', unsafe_allow_html=True)


def navigate(view):
    st.session_state["workspace_view"] = view


def quick_actions():
    for column, number, title, text, destination in zip(
        st.columns(3), ("01", "02", "03"),
        ("Compare detectors", "Replay traffic", "Analyze a file"),
        ("Find the balance between recall and false alarms.",
         "Follow threshold changes, one window at a time.",
         "Run a detector on your own compatible CSV."),
        ("Compare", "Replay", "Analyze CSV"),
    ):
        with column:
            with st.container(border=True):
                st.caption(number)
                st.markdown(f"**{title}**")
                st.caption(text)
                st.button(title + " →", key="go_" + destination,
                          on_click=navigate, args=(destination,), width="stretch")
