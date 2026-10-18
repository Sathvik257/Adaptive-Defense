from pathlib import Path
import hashlib
import pandas as pd
import streamlit as st
from adaptive_defense.predict import predict
from adaptive_defense.uploads import read_upload


def render(run, summary, metrics, pred):
    st.subheader("How the detector works")
    st.markdown('<div class="pipeline"><span>Traffic</span><b>→</b><span>Autoencoder</span><b>→</b><span>Anomaly score</span><b>→</b><span>Adaptive threshold</span><b>→</b><span>Alert</span></div>',unsafe_allow_html=True)
    st.write("DQN adjusts the threshold one grid step at a time using past window statistics. Development labels provide rewards after predictions. Final evaluation does not expose labels to the policy or update its weights.")
    st.write("The adapted model receives feedback only from a separate adaptation partition. Its evaluation restarts with no labels available.")
    for limitation in summary["limitations"]:
        st.write("• "+limitation)
    with st.expander("Experiment configuration"):
        st.json(summary)
