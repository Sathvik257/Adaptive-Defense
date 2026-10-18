from pathlib import Path
import hashlib
import pandas as pd
import streamlit as st
from adaptive_defense.predict import predict
from adaptive_defense.uploads import read_upload


def render(run, summary, metrics, pred):
    st.subheader("Watch the detector work")
    st.caption("Saved offline predictions; this is not a live network capture.")
    trace=pd.read_csv(run/"threshold_trace.csv")
    if st.session_state.get("replay_run") != str(run):
        st.session_state.update(replay_run=str(run),replay_play=False,replay_index=1)
    def toggle_play():
        if st.session_state.replay_index >= len(trace):
            st.session_state.replay_index=1
        st.session_state.replay_play=not st.session_state.replay_play
    def reset_replay():
        st.session_state.update(replay_play=False,replay_index=1)
    controls=st.columns([1,1,5])
    controls[0].button("Pause" if st.session_state.replay_play else "Play replay",on_click=toggle_play,type="primary")
    controls[1].button("Reset",on_click=reset_replay)
    @st.fragment(run_every="900ms" if st.session_state.replay_play else None)
    def replay_view():
        count=st.session_state.replay_index
        st.progress(count/len(trace),text=f"Window {count} of {len(trace)}")
        st.line_chart(trace.head(count)[["threshold"]],color="#347459")
        st.bar_chart(trace.head(count)[["alerts"]],color="#93ac99")
        end=int(trace.iloc[count-1]["end"])
        st.dataframe(pred.head(end).tail(30),width="stretch",hide_index=True)
        if st.session_state.replay_play:
            if count<len(trace):
                st.session_state.replay_index=count+1
            else:
                st.session_state.replay_play=False
                st.rerun()
    replay_view()
