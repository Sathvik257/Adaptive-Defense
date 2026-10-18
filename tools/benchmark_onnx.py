"""Export trusted AE weights and measure CPU model-only latency (not a full NIDS)."""
import argparse
import json
import pickle
import platform
import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adaptive_defense.data import clean_features
from adaptive_defense.models import Autoencoder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/demo')
    p.add_argument('--csv',default='runs/demo/synthetic_input.csv')
    p.add_argument('--output',required=True)
    p.add_argument('--iterations',type=int,default=200)
    p.add_argument('--threads',type=int,default=1)
    args=p.parse_args()
    if min(args.iterations,args.threads)<1:
        p.error('Iterations and threads must be positive.')
    out=Path(args.output)
    if out.exists():
        p.error('Choose a new output directory.')
    out.mkdir(parents=True)
    run=Path(args.run)
    with (run/'pipeline.pkl').open('rb') as f:
        bundle=pickle.load(f)
    model=Autoencoder(bundle['dimension']);model.eval()
    model.load_state_dict(torch.load(run/'autoencoder.pt',weights_only=True))
    x=np.asarray(bundle['preprocessor'].transform(clean_features(pd.read_csv(args.csv),bundle['columns'])),dtype=np.float32)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Finite nonempty input required.')
    torch.onnx.export(model,torch.from_numpy(x[:1]),str(out/'autoencoder.onnx'),
        input_names=['features'],output_names=['reconstruction'],opset_version=17,
        dynamic_axes={'features':{0:'batch'},'reconstruction':{0:'batch'}},dynamo=False)
    options=ort.SessionOptions();options.intra_op_num_threads=args.threads;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(out/'autoencoder.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        expected=model(torch.from_numpy(x[:32])).numpy()
    actual=session.run(None,{'features':x[:32]})[0]
    np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
    results=[]
    for batch_size in (1,32,256):
        batch=np.ascontiguousarray(x[:min(batch_size,len(x))])
        for _ in range(20):
            session.run(None,{'features':batch})
        durations=[]
        for _ in range(args.iterations):
            start=time.perf_counter_ns();session.run(None,{'features':batch})
            durations.append((time.perf_counter_ns()-start)/1e6)
        results.append({'batch_size':len(batch),'iterations':args.iterations,
                        'median_batch_ms':float(np.median(durations)),'p95_batch_ms':float(np.percentile(durations,95)),
                        'flows_per_second':len(batch)*1000/float(np.mean(durations))})
    report={'hardware':platform.processor(),'machine':platform.machine(),'platform':platform.platform(),
            'onnxruntime':ort.__version__,'provider':session.get_providers(),'threads':args.threads,
            'precision':'float32','parity_max_abs_error':float(np.abs(actual-expected).max()),
            'scope':'AE reconstruction only; excludes preprocessing, anomaly reduction, DQN, capture and API',
            'results':results}
    (out/'latency.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
