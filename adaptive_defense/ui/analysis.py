from pathlib import Path
import hashlib
import pandas as pd
import streamlit as st
from adaptive_defense.predict import predict
from adaptive_defense.uploads import read_upload


def render(run, summary, metrics, pred):
    st.subheader("Compare the trade-off")
    chosen=st.multiselect("Models to compare",metrics.model.tolist(),default=["Fixed autoencoder","DQN frozen","DQN adapted"])
    if chosen:
        comparison=metrics.set_index("model").loc[chosen]
        st.scatter_chart(comparison.reset_index(),x="false_positive_rate",y="recall",color="model",size=150)
        st.caption("Higher recall and fewer false alarms are preferable. These points describe this experiment only.")
    st.subheader("Measured model performance")
    st.dataframe(metrics,width="stretch",hide_index=True)
    st.download_button("Download metrics",(run/"metrics.csv").read_bytes(),"metrics.csv","text/csv")
    name=st.selectbox("Inspect confusion matrix",metrics.model)
    r=metrics.set_index("model").loc[name]
    st.dataframe(pd.DataFrame([[r.tn,r.fp],[r.fn,r.tp]],index=["Actually normal","Actually attack"],columns=["Predicted normal","Predicted attack"]))
    per_category=pred.groupby("attack_cat")[list(metrics.model)].mean()
    st.subheader("Alert fraction by category")
    st.dataframe(per_category.style.format("{:.1%}"),width="stretch")
    st.caption("For Normal this is the false positive rate; for attack categories this is recall.")
    st.subheader("Autoencoder learning")
    st.line_chart(pd.read_csv(run/"autoencoder_history.csv").set_index("epoch"))
    st.subheader("Controller learning")
    st.line_chart(pd.read_csv(run/"controller_history.csv").set_index("episode"))
    st.subheader("Inspect decisions")
    selected_category=st.selectbox("Traffic category",["All",*sorted(pred.attack_cat.unique())])
    decision=st.selectbox("Decision type",["All","Missed attacks","False alarms","Correct decisions"])
    records=pred.copy()
    if selected_category!="All":
        records=records[records.attack_cat==selected_category]
    if decision=="Missed attacks":
        records=records[(records.label==1)&(records[name]==0)]
    elif decision=="False alarms":
        records=records[(records.label==0)&(records[name]==1)]
    elif decision=="Correct decisions":
        records=records[records.label==records[name]]
    st.caption(f"{len(records):,} matching flows · displaying up to 200")
    st.dataframe(records[["attack_cat","label","anomaly_score",name]].head(200),hide_index=True,width="stretch")
    st.download_button("Download filtered decisions",records.to_csv(index=False),"filtered_decisions.csv","text/csv")
