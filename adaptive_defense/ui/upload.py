from pathlib import Path
import hashlib
import pandas as pd
import streamlit as st
from adaptive_defense.uploads import read_upload


def render(run, summary, metrics, pred):
    st.subheader("Analyze compatible flow records")
    st.write("Upload a CSV with the same feature columns as this experiment. Labels are optional and are excluded from inference.")
    st.caption("Up to 10 MB and 10,000 rows. Files are processed in server memory and are not saved by this app.")
    with st.expander("CSV format and example"):
        st.write("Required columns: "+", ".join(summary["feature_columns"]))
        sample=run/"synthetic_input.csv"
        if sample.exists():
            st.download_button("Download example CSV",pd.read_csv(sample,nrows=25)[summary["feature_columns"]].to_csv(index=False),"example_flows.csv","text/csv")
    file=st.file_uploader("Network flow CSV",type="csv")
    file_key=hashlib.sha256(file.getvalue()).hexdigest() if file else None
    mode=st.radio("Detector",["Fixed autoencoder","DQN frozen","DQN adapted"],horizontal=True)
    if file and st.button("Analyze traffic",type="primary"):
        try:
            frame=read_upload(file)
            modes={"Fixed autoencoder":"fixed","DQN frozen":"frozen","DQN adapted":"adapted"}
            from adaptive_defense.predict import predict
            with st.spinner("Analyzing traffic…"):
                output=predict(run,frame,modes[mode])
            st.session_state["analysis_result"]=(str(run),file_key,mode,output)
            alerts=(output.prediction=="Suspicious").to_numpy()
            st.success(f"Analyzed {len(output):,} flows; flagged {int(alerts.sum()):,}.")
        except (ValueError,TypeError,KeyError) as exc:
            st.error(str(exc))
    saved=st.session_state.get("analysis_result")
    if saved and saved[0]==str(run) and file and saved[1]==file_key and saved[2]==mode:
        st.dataframe(saved[3].head(500),width="stretch",hide_index=True)
        st.download_button("Download predictions",saved[3].to_csv(index=False),"flow_predictions.csv","text/csv")
