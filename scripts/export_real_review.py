"""Export non-audio observations from a completed private three-arm study."""
import argparse
import bisect
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def verify_record(record):
    """Independent scalar count oracle, without production diagnostic imports."""
    ledger={int(row[0]):tuple(map(int,row[1:])) for row in record['committed_ledger']}
    if len(ledger)!=len(record['committed_ledger']):raise AssertionError('Duplicate committed timestamp')
    exported=defaultdict(set)
    for t,lane in record['exported_events']:
        if lane not in range(4) or lane in exported[t]:raise AssertionError('Invalid or duplicate lane')
        exported[t].add(lane)
    timeline=sorted(set(ledger)|set(exported))
    total=defaultdict(int);included=set();errors=[];common_errors=[]
    names=('music','prepared','layout','export')
    for beat in record['beats']:
        stamps=timeline[bisect.bisect_left(timeline,beat['start_ms']):bisect.bisect_left(timeline,beat['end_ms'])]
        included.update(stamps)
        counts=[]
        for t in stamps:
            a,b,mask=ledger.get(t,(0,0,0))
            counts.append((a,b,mask.bit_count(),len(exported[t])))
        for i,name in enumerate(names):
            n=sum(row[i] for row in counts);r=sum(row[i]>0 for row in counts)
            if [n,r]!=beat['stage_counts'][name]:raise AssertionError('Per-beat stage count mismatch')
            total[name+'_notes']+=n;total[name+'_rows']+=r
        for i,name in enumerate(('prepare','layout','export')):
            losses={name+'_removed_notes':sum(max(0,row[i]-row[i+1]) for row in counts),
                    name+'_added_notes':sum(max(0,row[i+1]-row[i]) for row in counts),
                    name+'_dropped_rows':sum(row[i]>0 and row[i+1]==0 for row in counts)}
            for key,value in losses.items():
                if value!=beat['losses'][key]:raise AssertionError('Gross transition mismatch')
                total[key]+=value
        if beat['complete']:
            errors.append([abs(beat['stage_counts']['export'][i]-beat['desired'][i]) for i in (0,1)])
            if 'unshaped_desired' in beat:
                common_errors.append([abs(beat['stage_counts']['export'][i]-beat['unshaped_desired'][i]) for i in (0,1)])
    total['full_beats']=len(errors)
    total['unattributed_positions']=len(set(timeline)-included)
    total['unattributed_music_notes']=sum(ledger.get(t,(0,0,0))[0] for t in set(timeline)-included)
    total['unattributed_export_notes']=sum(len(exported[t]) for t in set(timeline)-included)
    total['notes_beat_mae']=math.fsum(e[0] for e in errors)/len(errors)
    total['rows_beat_mae']=math.fsum(e[1] for e in errors)/len(errors)
    for key,value in total.items():
        if abs(value-record['summary'][key])>1e-10:raise AssertionError('Summary mismatch '+key)
    bars=record['style_bars'];eligible=[b for b in bars if b['eligible']]
    for bar in eligible:
        if 'target_coverage' in bar:
            expected=bar['target_coverage']>=.5 and bar['largest_other_coverage']<.5
            if expected!=bar['pass']:raise AssertionError('Style threshold mismatch')
    passed=sum(b['pass'] for b in eligible)
    if len(eligible)!=record['outcome']['adherence']['eligible_bars'] or passed!=record['outcome']['adherence']['passed_bars']:
        raise AssertionError('Style numerator mismatch')
    if common_errors:
        means=[math.fsum(e[i] for e in common_errors)/len(common_errors) for i in (0,1)]
        if any(abs(a-b)>1e-10 for a,b in zip(means,record['outcome']['common_unshaped_mae'])):
            raise AssertionError('Common-target MAE mismatch')
    return {'full_beats':len(errors),'committed_positions':len(ledger),'exported_notes':total['export_notes'],
            'passed_bars':passed,'eligible_bars':len(eligible),'recalculated_mae':[total['notes_beat_mae'],total['rows_beat_mae']]}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    protocol=read(args.run/'protocol.json')
    results=read(args.run/('analysis-results.json' if (args.run/'analysis-results.json').is_file() else 'results.json'))
    paired=read(args.run/'paired-summary.json')
    if any(r['status']!='complete' for r in results.values()):raise AssertionError('Incomplete study')
    args.output.mkdir(parents=True,exist_ok=True)
    exported=[]
    for variant in protocol['variants']:
        folder=args.run/variant
        trace=read(folder/'density-execution.json')
        events=[]
        for line in (folder/'generated.osu').read_text(encoding='utf-8').split('[HitObjects]')[1].strip().splitlines():
            fields=line.split(',');events.append([int(fields[2]),min(3,int(fields[0])*4//512)])
        record={'variant':variant,'scope':'New frozen-feature neural inference on one known development song and one global seed. No source audio distributed.',
                'outcome':results[variant], 'summary':trace['summary'],
                'committed_ledger':read(folder/'committed-ledger.json'),'exported_events':events,
                'beats':[{k:b[k] for k in ['start_ms','end_ms','complete','desired','stage_counts','losses']}
                    for bar in trace['bars'] for b in bar['beats']],
                'style_bars':read(folder/'bar-adherence.json')['bars'],
                'source_record_sha256':{name:digest(folder/name) for name in ['generation.json','density-execution.json','committed-ledger.json','generated.osu','bar-adherence.json']}}
        unshaped=read(folder/'unshaped-targets.json')
        for i,beat in enumerate(record['beats']):beat['unshaped_desired']=unshaped[i//4][i%4]
        check=verify_record(record)
        target=args.output/f'current-real-{variant}.json'
        target.write_text(json.dumps(record,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
        exported.append({'variant':variant,'record':target.name,'sha256':digest(target),'independent_check':check})
    summary={'scope':'One known development song, one seed, frozen cached frames. Matched bars are dependent; no population inference.',
        'protocol':protocol,'outcomes':results,'paired':paired,'records':exported,
        'new_inference_completed':True,'audio_redistributed':False,'baseline_promoted':False,
        'analysis_script_sha256':digest(Path(__file__)),
        'execution_history':('Generated packages completed; initial post-analysis expected a loose chart that the pipeline cleans up. Original failure records retained privately; charts recovered from hash-verified original packages without regeneration.'
            if any(r.get('analysis_recovered_from_verified_package',False) for r in results.values()) else
            'All generated packages and initial analysis completed. Independent analysis rechecked the same hash-verified packages without regeneration.')}
    (args.output/'current-real-study.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'arms':len(exported),'paired':paired,'outcomes':{k:{f:v[f] for f in ['notes','rows','actual_star','adherence']} for k,v in results.items()}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
