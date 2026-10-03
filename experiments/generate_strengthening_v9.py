"""Generate full frozen trajectories on the prospectively locked new sources."""
from pathlib import Path
import os,json,hashlib,time,importlib.metadata as md
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from chronos import BaseChronosPipeline
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/strengthening_v9/forecasts'
LEVELS=[.1,.3,.5,.7,.9];H=24
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/strengthening_v9/manifest.json').read_text())
    protocol_sha=sha(ROOT/'docs/STRENGTHENING_PROTOCOL_V9.md');assert protocol_sha==manifest['protocol_sha256']
    for repo,label in [('amazon/chronos-bolt-base','Chronos-Boltenc-dec'),('amazon/chronos-2','Chronos-2enc-only')]:
        old=json.loads((ROOT/f'results/confirmation_v7/forecasts/{label}__traffic_c0.json').read_text())
        snapshot=Path(snapshot_download(repo,revision=old['model_snapshot'],local_files_only=True))
        hashes={p.name:sha(p) for p in snapshot.iterdir() if p.suffix in ('.json','.safetensors')}
        assert hashes==old['model_files']
        pipe=BaseChronosPipeline.from_pretrained(str(snapshot),device_map='cuda',dtype=torch.bfloat16)
        for entry in manifest['series']:
            src=ROOT/entry['path'];assert sha(src)==entry['sha256']
            name=entry['name'];dest=OUT/f'{label}__{name}.npz';meta=dest.with_suffix('.json')
            if dest.exists() and meta.exists():
                rec=json.loads(meta.read_text());assert rec['archive_sha256']==sha(dest) and rec['generator_sha256']==sha(__file__)
                continue
            with np.load(src) as z:y=z['y'];ctx=z['context_y'];times=z['time']
            origins=np.arange(600,len(y)-H);qout=np.empty((len(origins),H,5),np.float32);muout=np.empty((len(origins),H),np.float32)
            begin=time.perf_counter()
            for start in range(0,len(origins),128):
                idx=origins[start:start+128];contexts=[torch.tensor(ctx[t+1-512:t+1],dtype=torch.float32) for t in idx]
                with torch.inference_mode():q,mu=pipe.predict_quantiles(contexts,prediction_length=H,quantile_levels=LEVELS)
                if isinstance(q,(list,tuple)):
                    q=torch.stack([v.reshape(-1,H,5)[0] for v in q]);mu=torch.stack([v.reshape(-1,H)[0] for v in mu])
                qout[start:start+len(idx)]=q.float().cpu().numpy();muout[start:start+len(idx)]=mu.float().cpu().numpy()
            assert np.isfinite(qout).all() and np.isfinite(muout).all()
            target=origins[:,None]+np.arange(1,H+1)
            np.savez_compressed(dest,series=y,context_series=ctx,time=times,t_index=origins,target_index=target,y=y[target],observed=np.isfinite(y[target]),mu=muout,quantiles=qout,q_levels=LEVELS,backbone=label,dataset=name,horizons=np.arange(1,H+1))
            rec=dict(archive_sha256=sha(dest),source_sha256=sha(src),protocol_sha256=protocol_sha,generator_sha256=sha(__file__),model_repository=repo,model_snapshot=snapshot.name,model_files=hashes,context=512,batch=128,dtype='bfloat16',versions={p:md.version(p) for p in ['torch','numpy','transformers','chronos-forecasting']},device=torch.cuda.get_device_name(),elapsed_seconds=time.perf_counter()-begin)
            meta.write_text(json.dumps(rec,indent=2)+'\n');print(dest.name,len(origins),round(rec['elapsed_seconds'],2),flush=True)
        del pipe;torch.cuda.empty_cache()
if __name__=='__main__':main()
