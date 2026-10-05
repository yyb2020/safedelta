"""Derive manuscript summaries exclusively from recomputed panel-level results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);a=p.parse_args();d=a.input
    panels=pd.read_csv(d/'criterion_predicts_transfer.csv')
    deep=panels[panels.setting=='Plate 7 native (400 cells)'];shallow=panels[panels.setting=='Plate 2 (20 cells)']
    grid=np.linspace(-.05,.10,151)
    def realized(g,t):return np.where(g.evid_split.to_numpy()>t,g.true_gain.to_numpy(),0).mean()
    rows=[]
    for c in sorted(deep.ctx.unique()):
        train=deep[deep.ctx!=c];test=deep[deep.ctx==c];threshold=grid[np.argmax([realized(train,t) for t in grid])]
        rows.append(dict(held=c,t_fitted=threshold,gain_fitted=realized(test,threshold),gain_t0=realized(test,0.),gain_always=test.true_gain.mean(),gain_never=0.))
    pd.DataFrame(rows).round(5).to_csv(d/'criterion_threshold_loco_v2.csv',index=False)
    deep_threshold=grid[np.argmax([realized(deep,t) for t in grid])]
    rows=[]
    for g,t,rule,setting in [(deep,0.,'sign rule t=0','Plate 7 native (400)'),(shallow,0.,'sign rule t=0','Plate 2 (20)'),(shallow,deep_threshold,'tuned on deep, applied to shallow','Plate 2 (20)')]:
        released=g.evid_split>t;positive=g.true_gain>0;negative=g.true_gain<0
        rows.append(dict(k=5,rule=rule,released=int(released.sum()),beneficial=int((released&positive).sum()),harmful=int((released&negative).sum()),abstained=int((~released).sum()),missed_beneficial=int((~released&positive).sum()),realized=round(float(realized(g,t)),5),always=round(float(g.true_gain.mean()),5),setting=setting))
    pd.DataFrame(rows).to_csv(d/'release_rule_validation.csv',index=False)
    if (d/'criterion_external_panels.csv').exists():
        ext=pd.read_csv(d/'criterion_external_panels.csv');rows=[]
        for name,g in ext[ext.base=='pooled_B'].groupby('dataset'):
            rel=g.evid_loo>0;rows.append(dict(ds=name,n=len(g),always=g.true_gain.mean(),gated=np.where(rel,g.true_gain,0).mean(),rel=rel.mean(),shared=(g.evid_shared>0).mean()))
        for g,name in [(shallow,'Tahoe plate 2'),(deep,'Tahoe plate 7')]:
            rel=g.evid_split>0;rows.append(dict(ds=name,n=len(g),always=g.true_gain.mean(),gated=np.where(rel,g.true_gain,0).mean(),rel=rel.mean(),shared=np.nan))
        pd.DataFrame(rows).sort_values('always').reset_index(drop=True).to_csv(d/'fig1d_external_validation.csv',index=False)
if __name__=='__main__':main()
