"""Export publication-friendly PNG charts from measured experiment artifacts."""
import argparse
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def export(run):
    run=Path(run)
    out=run/"figures"
    out.mkdir(exist_ok=True)
    summary=json.loads((run/"summary.json").read_text())
    label="SYNTHETIC DEMO — not real attack evidence" if "SYNTHETIC" in summary["dataset"] else "Offline custom-split evaluation"
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    metrics=pd.read_csv(run/"metrics.csv").set_index("model")
    ax=metrics[["recall","precision","false_positive_rate"]].plot.bar(figsize=(11,6),color=["#0284c7","#0d9488","#e87936"])
    ax.set_ylim(0,1.05); ax.set_ylabel("Fraction"); ax.set_xlabel("")
    ax.set_title("Model comparison\n"+label)
    ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),frameon=False)
    plt.xticks(rotation=20,ha="right"); plt.tight_layout()
    plt.savefig(out/"model_comparison.png",dpi=180); plt.close()
    trace=pd.read_csv(run/"threshold_trace.csv")
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    axes[0].plot(trace.threshold,color="#0284c7"); axes[0].set_ylabel("Threshold")
    axes[1].bar(np.arange(len(trace)),trace.alerts,color="#0d9488"); axes[1].set_ylabel("Alerts")
    axes[1].set_xlabel("Synthetic replay window")
    fig.suptitle("DQN frozen controller\n"+label)
    fig.tight_layout(); fig.savefig(out/"threshold_replay.png",dpi=180); plt.close(fig)
    history=pd.read_csv(run/"autoencoder_history.csv")
    history.set_index("epoch").plot(figsize=(9,5))
    plt.ylabel("Mean squared error"); plt.title("Benign autoencoder learning\n"+label)
    plt.tight_layout(); plt.savefig(out/"autoencoder_learning.png",dpi=180); plt.close()
    print(f"Exported figures to {out}")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    export(p.parse_args().run)
