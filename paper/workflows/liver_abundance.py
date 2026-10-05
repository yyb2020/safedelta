"""Raw-rebuilt spot scores + official deconvolution metadata -> abundance check."""
import argparse,json
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr,mannwhitneyu
from liver_raw import regions


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata-dir',type=Path,required=True);a=p.parse_args()
    scores=pd.read_csv(a.output/'liver_spatial_spot_scores.csv');meta=regions(a.metadata_dir)
    props=['cell2loc_hep_prop','cell2loc_stellate_prop','cell2loc_chol_prop']
    M=scores.drop(columns=props).merge(meta[['section','barcode']+props],on=['section','barcode'],how='inner',validate='one_to_one')
    if M[props].isna().any().any():raise ValueError('missing official deconvolution covariates')
    M2=M.assign(bin=pd.qcut(M.cell2loc_stellate_prop,5,labels=False));rows=[]
    for name in ['core94_h','Bdom_h','Ddom_h']:
        table=M2.groupby(['bin','region'])[name].median().unstack()
        for b in table.index:
            rows.append(dict(score=name,stellate_quintile=int(b),stellate_prop_median=float(M2.loc[M2.bin==b,'cell2loc_stellate_prop'].median()),n_scar=int(((M2.bin==b)&(M2.region=='Scar')).sum()),n_hep=int(((M2.bin==b)&(M2.region=='Hep')).sum()),scar_median=float(table.loc[b,'Scar']),hep_median=float(table.loc[b,'Hep']),scar_minus_hep=float(table.loc[b,'Scar']-table.loc[b,'Hep']),rho_with_stellate=float(spearmanr(M.cell2loc_stellate_prop,M[name])[0])))
    pd.DataFrame(rows).round(6).to_csv(a.output/'liver_spatial_state_vs_abundance.csv',index=False)
    scar=M2[(M2.bin==4)&(M2.region=='Scar')];hep=M2[(M2.bin==4)&(M2.region=='Hep')]
    result={'top_quintile_scar_spots':len(scar),'top_quintile_hep_spots':len(hep),'core_score_scar_minus_hep':float(scar.core94_h.median()-hep.core94_h.median()),'residual_score_scar_minus_hep':float(scar.Ddom_h.median()-hep.Ddom_h.median()),'mann_whitney_one_sided_p':float(mannwhitneyu(scar.core94_h,hep.core94_h,alternative='greater').pvalue),'inference_unit':'spot; reproduction of the reported number does not establish independence of clustered spots'}
    (a.output/'liver_abundance_claims.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
