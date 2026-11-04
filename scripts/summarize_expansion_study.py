"""Analyze every retained expanded-study arm and export non-audio evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protocol=read(a.run/'protocol.json');executions=read(a.run/'execution-results.json')
    if len(executions)!=12 or any(r['status']!='complete' for r in executions):raise AssertionError('All 12 fixed-protocol outcomes required')
    a.output.mkdir(parents=True,exist_ok=True)
    cases=[]
    for track in protocol['tracks']:
        tid=track['track_id']
        freeze=read(a.run/tid/'frozen-upstream-check.json')
        if not freeze['all_cached_and_inferred_arrays_bitwise_equal']:raise AssertionError('Upstream cache equality failed')
        for seed in protocol['seeds']:
            run=a.run/tid/f'seed-{seed}';dest=a.output/f'{tid}-seed-{seed}'
            for script,arguments in [('analyze_real_song_run.py',['--run',str(run)]),
                                     ('export_real_review.py',['--run',str(run),'--output',str(dest)])]:
                proc=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'scripts'/script),*arguments],capture_output=True,text=True,encoding='utf-8')
                (run/(script+'.analysis.txt')).write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
                if proc.returncode:raise RuntimeError('Analysis failed; see retained private analysis transcript')
            d=read(dest/'current-real-study.json')
            if not d['paired']['exact_inferred_intermediate_gate_passed']:raise AssertionError('Inferred score equality failed')
            full=d['outcomes']['hierarchy-shaped'];only=d['outcomes']['hierarchy-only'];legacy=d['outcomes']['legacy']
            primary=d['paired']['matched_style_comparisons']['hierarchy-only__hierarchy-shaped']
            secondary=d['paired']['matched_style_comparisons']['legacy__hierarchy-shaped']
            hashes={}
            for variant in protocol['variants']:
                record=read(dest/f'current-real-{variant}.json')
                hashes[variant]=hashlib.sha256(json.dumps(sorted(record['exported_events']),separators=(',',':')).encode()).hexdigest()
            cases.append({'track_id':tid,'seed':seed,'study':(dest/'current-real-study.json').relative_to(a.output).as_posix(),
                'study_sha256':hashlib.sha256((dest/'current-real-study.json').read_bytes()).hexdigest(),
                'upstream_scores_bitwise_equal':True,'outcomes':d['outcomes'],
                'primary_shaping_comparison':{'common_note_mae_delta':full['common_unshaped_mae'][0]-only['common_unshaped_mae'][0],
                    'common_row_mae_delta':full['common_unshaped_mae'][1]-only['common_unshaped_mae'][1],
                    'matched':primary},
                'full_vs_legacy':{'common_note_mae_delta':full['common_unshaped_mae'][0]-legacy['common_unshaped_mae'][0],
                    'common_row_mae_delta':full['common_unshaped_mae'][1]-legacy['common_unshaped_mae'][1],
                    'own_plan_pass_count_delta':full['adherence']['passed_bars']-legacy['adherence']['passed_bars'],
                    'matched':secondary},'canonical_event_sha256':hashes})
    seed_checks=[]
    for t in protocol['tracks']:
        tc=[c for c in cases if c['track_id']==t['track_id']]
        seed_checks.append({'track_id':t['track_id'],'different_event_sets_by_variant':{v:tc[0]['canonical_event_sha256'][v]!=tc[1]['canonical_event_sha256'][v] for v in protocol['variants']}})
    result={'scope':'Two development excerpts and two decoder/global seeds, nested observations. One fresh frozen feature extraction per track; upstream predictions cached across arms and seeds.',
        'protocol':protocol,'cases':cases,'seed_variation':seed_checks,'neural_arms_completed':12,
        'all_upstream_scores_bitwise_equal':True,'audio_redistributed':False,'baseline_promoted':False,
        'sampling_boundary':'Eligibility and SHA order chosen before generation; raw sources and source rights remain private/unresolved. No held-out or population inference.',
        'runner_source_sha256':hashlib.sha256((ROOT/'scripts/run_expansion_study.py').read_bytes()).hexdigest(),
        'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if result['runner_source_sha256']!=protocol['runner_sha256']:raise AssertionError('Runner changed during study')
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (a.output/'outcomes.csv').open('w',encoding='utf-8',newline='') as stream:
        fields=['track_id','seed','variant','notes','rows','actual_star','passed_bars','eligible_bars','own_note_mae','common_note_mae','common_row_mae']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for case in cases:
            for variant,outcome in case['outcomes'].items():
                writer.writerow({'track_id':case['track_id'],'seed':case['seed'],'variant':variant,
                    **{k:outcome[k] for k in ['notes','rows','actual_star']},
                    'passed_bars':outcome['adherence']['passed_bars'],'eligible_bars':outcome['adherence']['eligible_bars'],
                    'own_note_mae':outcome['execution']['notes_beat_mae'],
                    'common_note_mae':outcome['common_unshaped_mae'][0],'common_row_mae':outcome['common_unshaped_mae'][1]})
    print(json.dumps({'arms':12,'seed_variation':seed_checks,'comparisons':[{k:c[k] for k in ['track_id','seed','primary_shaping_comparison','full_vs_legacy']} for c in cases]},indent=2))


if __name__=='__main__':main()
