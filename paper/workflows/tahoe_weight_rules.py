"""Execute recovered weight-rule analyses on freshly raw-rebuilt 16-line arrays.

The source-only 15-context mixture is a separate experiment. These scripts retain
the original 16-line candidate definitions, scalar grids, rounding and seeds.
"""
import argparse
import hashlib,json,time
import os
from pathlib import Path
import runpy
import sys
import numpy as np

SCRIPTS=['weightrule2','weightrule','lib_ablate','headroom','improve']

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--analyses',nargs='+',choices=SCRIPTS,default=SCRIPTS[:3]);a=p.parse_args();out=a.output.resolve()
    cube=out/'plate7_16line_pseudobulk.npz'
    with np.load(cube,allow_pickle=False) as z:
        if z['deep'].shape!=(2816,7007) or z['lines'].shape!=(16,) or z['conds'].shape!=(86,):raise ValueError('wrong reconstruction; expected archived 16-line × 86-condition × 7007-gene contract')
    source=Path(__file__).resolve().parents[1]/'legacy_sources';os.chdir(out)
    receipt={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'cube_sha256':hashlib.sha256(cube.read_bytes()).hexdigest(),'analyses':[]}
    for name in a.analyses:
        receipt['analyses'].append({'script':name+'.py','sha256':hashlib.sha256((source/(name+'.py')).read_bytes()).hexdigest()})
        print('Starting',name,flush=True);sys.argv=[name+'.py',str(cube)];runpy.run_path(str(source/(name+'.py')),run_name='__main__')
        receipt['analyses'][-1]['completed']=True
        (out/'weight_rule_run.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
