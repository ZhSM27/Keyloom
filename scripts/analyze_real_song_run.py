"""Analyze retained study packages without regenerating any model output."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as s:
        for block in iter(lambda:s.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    from keyloom_mania.main_style import STYLE_NAMES,evaluate_bars
    from keyloom_mania.patterns import analyze_file
    protocol=read(args.run/'protocol.json');attempts=read(args.run/'results.json');results={}
    for variant in protocol['variants']:
        folder=args.run/variant;manifest=read(folder/'generation.json');trace=read(folder/'density-execution.json')
        package=next(folder.glob('*.osz'))
        if sha(package)!=manifest['package_sha256']:raise AssertionError('Package changed since generation')
        with zipfile.ZipFile(package) as z:
            chart=z.read(next(n for n in z.namelist() if n.endswith('.osu')))
        target=folder/'generated.osu'
        if target.exists() and target.read_bytes()!=chart:raise AssertionError('Existing analysis chart differs from package')
        if not target.exists():target.write_bytes(chart)
        bars=read(folder/'actual-bars.json')
        starts=np.array([b['start_ms'] for b in bars]);ends=np.array([b['end_ms'] for b in bars])
        ids=np.array([STYLE_NAMES.index(b['style']) if b['style'] else 0 for b in bars])
        complete=ends-starts>=4*60000/protocol['settings']['bpm']-1
        adherence=evaluate_bars(analyze_file(target),starts,ends,ids,np.array([b['quiet'] for b in bars]),complete)
        (folder/'bar-adherence.json').write_text(json.dumps(adherence,ensure_ascii=False,indent=2),encoding='utf-8')
        with np.load(folder/'planner-inputs.npz') as a:
            original=a['unshaped_beat_density'];shaped=a['beat_density']
            (folder/'unshaped-targets.json').write_text(json.dumps(original.tolist()),encoding='utf-8')
            target_totals={'unshaped_notes':float(original[...,0].sum(dtype=np.float64)),
                'shaped_notes':float(shaped[...,0].sum(dtype=np.float64)),
                'shaped_rows':float(shaped[...,1].sum(dtype=np.float64))}
            errors=[]
            for i,bar in enumerate(trace['bars']):
                for j,beat in enumerate(bar['beats']):
                    if beat['complete']:errors.append(abs(np.array(beat['stage_counts']['export'])-original[i,j]))
        results[variant]={'status':'complete','generation_status':'complete',
            'analysis_recovered_from_verified_package':attempts[variant]['status']!='complete',
            'notes':manifest['notes'],'rows':manifest['rows'],'actual_star':manifest['actual_star'],
            'sections':manifest['hierarchy_sections'],'phrases':manifest['hierarchy_phrases'],
            'execution':trace['summary'],'density':read(folder/'bar-density.json')['summary'],
            'common_unshaped_mae':np.mean(errors,axis=0).tolist(),'target_totals':target_totals,
            'adherence':{k:adherence[k] for k in ['eligible_bars','passed_bars','bar_pass_rate','per_style']},
            'identity':{k:manifest.get(k) for k in ['generator_sha256','arrangement_sha256','density_sha256','inference_policy_sha256','generator_runtime_policy']},
            'chart_sha256':sha(target),'elapsed_seconds':attempts[variant]['elapsed_seconds']}
    identities=[r['identity'] for r in results.values()]
    if not all(x==identities[0] for x in identities):raise AssertionError('Unequal model or policy identity')
    intermediate_audit={}
    with np.load(args.run/'legacy/planner-inputs.npz') as a:
        for variant in protocol['variants'][1:]:
            with np.load(args.run/variant/'planner-inputs.npz') as b:
                for key in ['music','acoustic','rhythm','base_density','unshaped_beat_density']:
                    np.testing.assert_array_equal(a[key],b[key])
                intermediate_audit[variant]={}
                for key in ['owner_scores','style_evidence']:
                    intermediate_audit[variant][key]={'bitwise_equal':bool(np.array_equal(a[key],b[key])),
                        'different_elements':int(np.count_nonzero(a[key]!=b[key])),
                        'max_absolute_difference':float(np.max(abs(a[key].astype(np.float64)-b[key].astype(np.float64))))}
    matched={}
    for aa,bb in [('legacy','hierarchy-only'),('hierarchy-only','hierarchy-shaped'),('legacy','hierarchy-shaped')]:
        a=read(args.run/aa/'bar-adherence.json')['bars'];b=read(args.run/bb/'bar-adherence.json')['bars']
        pairs=[(x,y) for x,y in zip(a,b) if x['eligible'] and y['eligible'] and x['planned_style']==y['planned_style']]
        matched[aa+'__'+bb]={'bars':len(pairs),'passed_a':sum(x['pass'] for x,y in pairs),'passed_b':sum(y['pass'] for x,y in pairs),
            'rate_a':float(np.mean([x['pass'] for x,y in pairs])) if pairs else None,
            'rate_b':float(np.mean([y['pass'] for x,y in pairs])) if pairs else None}
    (args.run/'analysis-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    exact=all(item['bitwise_equal'] for arm in intermediate_audit.values() for item in arm.values())
    (args.run/'paired-summary.json').write_text(json.dumps({'identical_frozen_inputs_and_models':True,
        'inferred_intermediate_audit':intermediate_audit,
        'exact_inferred_intermediate_gate_passed':exact,
        'initial_exact_intermediate_gate':'Passed for owner/style inferred scores.' if exact else 'Failed for owner/style inferred scores; retained as a limitation, not silently relaxed to bitwise equality.',
        'matched_style_comparisons':matched,
        'boundary':'One known song and one seed, descriptive native-rule audit; dependent bars, density-responsive roles and small GPU inferred-score drift. No bitwise-clamped causal, music-quality or population claim.'},indent=2),encoding='utf-8')
    print(json.dumps({'outcomes':{k:{x:r[x] for x in ['notes','rows','actual_star','adherence','common_unshaped_mae','target_totals']} for k,r in results.items()},'matched':matched},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
