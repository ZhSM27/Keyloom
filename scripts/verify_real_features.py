"""Low-memory planning checks on a retained real-audio feature/target record.

This recalculates planning only. It neither trains nor executes a generator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keyloom_mania.hierarchical_planner import section_phrase_structure, shape_hierarchical_density, json_hierarchy


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--bar-record',type=Path,required=True)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with np.load(args.inputs,allow_pickle=False) as data:
        music,acoustic,rhythm=[data[k] for k in ['music','acoustic','rhythm']]
        density=data['unshaped_beat_density'].copy()
    bars=json.loads(args.bar_record.read_text(encoding='utf-8'))['bars']
    quiet=np.array([b['quiet'] for b in bars])
    h=section_phrase_structure(quiet,music=music,acoustic=acoustic,rhythm=rhythm)
    shaped,report=shape_hierarchical_density(density,h)
    w=density[...,0].sum(-1).astype(np.float64)
    active=(~quiet)&(w>0)
    q=np.ones(len(density))
    for s in h['sections']:
        q[s['start_bar']:s['end_bar']]+=.08*np.clip(s['energy_level'],-1,1)
    for phrase in h['segments']:
        a,b=phrase['start_bar'],phrase['end_bar'];z=h['energy_z'][a:b]
        q[a:b]+=.04*np.clip(z-z.mean(),-1,1)
    normalized=q.copy()
    normalized[active]*=w[active].sum()/(q[active]*w[active]).sum()
    variants={'identity':density,'unrenormalized_contrast':density*q[:,None,None],
              'unbounded_renormalized_contrast':density*normalized[:,None,None],'hierarchical_bounded':shaped}
    results={}
    for name,value in variants.items():
        factors=np.ones(len(w));factors[active]=value[...,0].sum(-1)[active]/w[active]
        results[name]={'notes_budget':float(value[...,0].sum(dtype=np.float64)),
            'rows_budget':float(value[...,1].sum(dtype=np.float64)),
            'note_budget_absolute_error':float(abs(value[...,0].sum(dtype=np.float64)-density[...,0].sum(dtype=np.float64))),
            'capacity_violating_beats':int(np.count_nonzero((value[...,0]>96+1e-5)|(value[...,1]>24+1e-5))),
            'max_relative_change':float(np.max(abs(factors-1))),
            'changed_bars':int(np.count_nonzero(abs(factors-1)>1e-6))}
    check=results['hierarchical_bounded']
    if check['note_budget_absolute_error']>1e-3 or check['capacity_violating_beats']:
        raise AssertionError('Planning invariant failed; retain result for investigation.')
    record={'scope':'Current planning algorithm applied to retained features/targets of one previously used development song. No current generator execution or unseen-song evidence.',
        'source_audio_sha256':args.source_sha256,'inputs_sha256':sha(args.inputs),'bar_record_sha256':sha(args.bar_record),
        'implementation_sha256':sha(ROOT/'keyloom_mania/hierarchical_planner.py'),'script_sha256':sha(__file__),
        'bars':len(bars),'sections':len(h['sections']),'phrases':len(h['segments']),
        'results':results,'projection_inputs':{'original_density':density.tolist(),'desired_multipliers':q.tolist(),
             'quiet':quiet.tolist(),'reported_multipliers':report['multipliers'],'reported_shaped_density':shaped.tolist()},
        'intervals':{'sections':[{k:s[k] for k in ['start_bar','end_bar','phrase_ids']} for s in h['sections']],
            'phrases':[{k:s[k] for k in ['start_bar','end_bar','section_id']} for s in h['segments']]},
        'boundary':'No audio or raw MusicFM/acoustic/rhythm features included. Training overlap and source rights unresolved. Projection is recalculable; hierarchy feature extraction is not reproducible from this extract.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:record[k] for k in ['bars','sections','phrases','results']},indent=2))


if __name__=='__main__':
    main()
