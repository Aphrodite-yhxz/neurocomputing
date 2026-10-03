"""Additional full author history-450 run; earlier history-600 output is preserved."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,subprocess,json,hashlib,time,os,argparse,shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from daci.full_horizon import to_target_time
OUT=ROOT/'results/focused_v10/acmcp_n450'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def job(p,rscript,force):
    dest=OUT/p.stem;dest.mkdir(parents=True,exist_ok=True)
    manifest=dict(archive_sha256=sha(p),author_acmcp_sha256=sha(ROOT/'docs/references/v4/acmcp/R/acmcp.R'),protocol_sha256=sha(ROOT/'docs/FOCUSED_FINISH_PROTOCOL_V10.md'))
    if (dest/'complete.json').exists() and not force:
        saved=json.loads((dest/'complete.json').read_text())
        for k,v in manifest.items():assert saved[k]==v
        return p.name,'cached'
    with np.load(p) as z:y=z['y'];mu=z['mu']
    assert np.isfinite(y).all()
    pf,errors=to_target_time(mu,y)
    np.savetxt(dest/'mean.csv',pf,delimiter=',');np.savetxt(dest/'errors.csv',errors,delimiter=',')
    cfg=dict(ncal=450,lr=.01,KI=max(float(np.nanmax(abs(errors[:600]))),1e-12),Csat=.5,scorecast=1,ma_method='CSS')
    pd.DataFrame([cfg]).to_csv(dest/'config.csv',index=False)
    wrapper=dest/'run_author_snapshot.R';wrapper.write_bytes((ROOT/'scripts/run_acmcp_author.R').read_bytes())
    manifest.update(configuration=cfg,r_wrapper_sha256=sha(wrapper))
    (dest/'started.json').write_text(json.dumps(manifest,indent=2)+'\n')
    begin=time.perf_counter()
    with (dest/'execution.log').open('w') as log:
        subprocess.run([rscript,'--vanilla',str(wrapper),str(ROOT),str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=0x08000000 if os.name=='nt' else 0)
    manifest['elapsed_seconds']=time.perf_counter()-begin
    (dest/'complete.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return p.name,manifest['elapsed_seconds']
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--rscript');ap.add_argument('--workers',type=int,default=4);ap.add_argument('--force',action='store_true');args=ap.parse_args()
    local=ROOT/'.tools/R-4.6.1/bin/Rscript.exe';rscript=args.rscript or (str(local) if local.exists() else shutil.which('Rscript'))
    if not rscript:raise SystemExit('Install R and forecast; pass --rscript PATH.')
    paths=sorted((ROOT/'results/strengthening_v9/forecasts').glob('*__weather_*.npz'));assert len(paths)==4
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(job,p,rscript,args.force) for p in paths]):print(f.result(),flush=True)
