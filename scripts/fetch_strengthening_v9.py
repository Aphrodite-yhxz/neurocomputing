"""Acquire the metadata-selected new sources only after protocol lock."""
from pathlib import Path
import hashlib,io,json,zipfile,requests
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/strengthening_v9'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=json.loads((ROOT/'results/strengthening_v9/protocol_lock.json').read_text())
    assert sha(ROOT/'docs/STRENGTHENING_PROTOCOL_V9.md')==lock['protocol_sha256']
    OUT.mkdir(parents=True,exist_ok=True);sources=[];series=[]
    for name,url in [('weather','https://weather.bgc-jena.mpg.de/mpi_roof.zip'),('air','https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip')]:
        p=OUT/(name+'.zip')
        if not p.exists():
            r=requests.get(url,timeout=90);r.raise_for_status();p.write_bytes(r.content)
        sources.append(dict(name=name,url=url,sha256=sha(p),bytes=p.stat().st_size))
        with zipfile.ZipFile(p) as z:
            print(name,z.namelist(),flush=True)
            if name=='weather':
                files=[n for n in z.namelist() if n.lower().endswith('.csv')];assert len(files)==1
                df=pd.read_csv(io.BytesIO(z.read(files[0])),encoding='latin1')
                print('Weather columns:',list(df.columns),flush=True)
                ts=pd.to_datetime(df.iloc[:,0],dayfirst=True)
                frame=df.set_index(ts).replace(-9999,np.nan)
                times=pd.date_range('2026-01-01',periods=3600,freq='h')
                for col,label in [('T (degC)','temperature'),('p (mbar)','pressure')]:
                    vals=pd.to_numeric(frame[col],errors='raise')
                    mean=vals.resample('h').mean().reindex(times)
                    count=vals.resample('h').count().reindex(times)
                    mean[count!=6]=np.nan
                    series.append((f'weather_{label}',mean.to_numpy(),times+pd.Timedelta(hours=1)))
            else:
                nested=[n for n in z.namelist() if n.lower().endswith('.zip')]
                if nested:z=zipfile.ZipFile(io.BytesIO(z.read(nested[0])))
                for station in ['Aotizhongxin','Dingling']:
                    f=[n for n in z.namelist() if station in n and n.endswith('.csv')];assert len(f)==1
                    df=pd.read_csv(io.BytesIO(z.read(f[0])))
                    ts=pd.to_datetime(df[['year','month','day','hour']])
                    times=pd.date_range('2013-03-01',periods=3600,freq='h')
                    vals=pd.Series(df['PM2.5'].to_numpy(),index=ts).reindex(times).to_numpy()
                    series.append((f'air_{station.lower()}',vals,times))
    records=[]
    for name,y,times in series:
        assert len(y)==3600
        filled=pd.Series(y).ffill().fillna(0).to_numpy()
        p=OUT/(name+'.npz');np.savez_compressed(p,y=y,context_y=filled,observed=np.isfinite(y),time=times.astype(str).to_numpy(dtype='U'))
        records.append(dict(name=name,path=p.relative_to(ROOT).as_posix(),sha256=sha(p),n=len(y),missing=int(np.isnan(y).sum()),first=str(times[0]),last=str(times[-1])))
    (OUT/'manifest.json').write_text(json.dumps(dict(protocol_sha256=lock['protocol_sha256'],sources=sources,series=records),indent=2)+'\n')
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
