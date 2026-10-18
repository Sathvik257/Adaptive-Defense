import pandas as pd
import streamlit as st
from adaptive_defense.ui.components import quick_actions


def render(run, summary, metrics, pred):
    st.subheader("Detection overview")
    selected=st.selectbox("Model",metrics.model)
    row=metrics.set_index("model").loc[selected]
    reference=metrics.set_index("model").loc["Fixed autoencoder"]
    a,b,c,d=st.columns(4)
    def delta(value, baseline, percent=True):
        if abs(value-baseline)<1e-8:
            return None
        return f"{(value-baseline)*100:+.1f} pp" if percent else f"{value-baseline:+.3f}"
    a.metric("Attack recall",f"{row.recall:.1%}",delta(row.recall,reference.recall))
    b.metric("Precision",f"{row.precision:.1%}",delta(row.precision,reference.precision))
    c.metric("False alarm rate",f"{row.false_positive_rate:.1%}",delta(row.false_positive_rate,reference.false_positive_rate),delta_color="inverse")
    d.metric("F1 score",f"{row.f1:.3f}",delta(row.f1,reference.f1,False))
    st.caption("Changes are relative to the fixed-threshold autoencoder. pp = percentage points.")
    st.markdown(f'<div class="insight">Of <strong>{int(row.tp+row.fn):,} attacks</strong>, this detector caught <strong>{int(row.tp):,}</strong> and missed <strong>{int(row.fn):,}</strong>. It also flagged <strong>{int(row.fp):,}</strong> normal flows.</div>',unsafe_allow_html=True)
    st.subheader("Detection at a glance")
    chart_data=metrics.set_index("model")[["known_recall","unseen_recall","false_positive_rate"]].rename(
        columns={"known_recall":"Known attack recall","unseen_recall":"Held-out category recall","false_positive_rate":"False alarm rate"})
    st.bar_chart(chart_data,stack=False,color=["#347459","#93ac99","#c69861"])
    st.caption("For DQN adapted, the held-out-category column measures post-feedback recall.")
    left,right=st.columns(2)
    with left:
        st.subheader("Experiment partitions")
        st.dataframe(pd.Series(summary["audit"]["split_counts"],name="Flows"),width="stretch")
    with right:
        st.subheader("Run details")
        st.write("Dataset:",summary["dataset"])
        st.write("Held-out categories:",", ".join(summary["audit"]["held_out_categories"]))
        st.write("Random seed:",summary["seed"])
        st.write("Execution time:",f"{summary['elapsed_seconds']:.1f} seconds")
    st.subheader("Explore the workspace")
    quick_actions()
