"""Offline matched development-song comparison using frozen audio features.

Audio and packages remain in the requested private output directory. No service
is started and no model defaults are changed. This is not an unseen-song test.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def packaged_chart(directory, expected_sha256):
    """Recover the exact validated chart after pipeline cleanup."""
    packages=list(directory.glob('*.osz'))
    if len(packages)!=1 or sha(packages[0])!=expected_sha256:
        raise ValueError('Expected one unchanged generated package.')
    with zipfile.ZipFile(packages[0]) as archive:
        charts=[n for n in archive.namelist() if n.endswith('.osu')]
        if len(charts)!=1:raise ValueError('Expected one chart in the package.')
        content=archive.read(charts[0])
    target=directory/'generated.osu'
    if target.exists() and target.read_bytes()!=content:
        raise ValueError('Existing analysis chart differs from the generated package.')
    if not target.exists():target.write_bytes(content)
    return target


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--rhythm', type=Path, required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bpm', type=float, required=True)
    p.add_argument('--offset', type=float, required=True)
    p.add_argument('--star', type=float, default=5.6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prepare-only', action='store_true', help='Freeze the intended matched design without importing the inference runtime')
    args = p.parse_args()
    if sha(args.audio) != args.source_sha256:
        raise ValueError('Source audio hash does not match the frozen protocol.')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.prepare_only:
        planned = {'scope':'Planned development comparison; no neural-generation result.',
            'source_sha256':args.source_sha256, 'feature_cache_sha256':sha(args.features), 'rhythm_cache_sha256':sha(args.rhythm),
            'seed':args.seed,'decoder_rng_policy':'Checkpoint-specific decoder seeds remain unchanged; effective policy is logged during execution.',
            'bpm':args.bpm,'offset_ms':args.offset,'target_star':args.star,
            'generator':'music-base-r1','role':'standard','density':'contrast','coordination':'coordinated',
            'primary_style':'Jumpstream','secondary_styles':['Stream','Rolls'],
            'variants':['legacy','hierarchy-only','hierarchy-shaped'],
            'input_policy':'Reuse identical frozen frame features in every arm.',
            'style_evaluator':'Native public OMTK intervals, not the historical grid-stream overlay.',
            'failure_policy':'Preserve attempted arms and failures without changing settings.',
            'source_sha256_by_file':{f:sha(ROOT/f) for f in ['scripts/review_real_song.py','keyloom_mania/studio/pipeline.py','keyloom_mania/hierarchical_planner.py']}}
        (args.output/'planned-protocol.json').write_text(json.dumps(planned,indent=2),encoding='utf-8')
        print('Frozen planned protocol. No inference executed.',flush=True)
        return
    import torch
    from keyloom_mania.studio import pipeline
    from keyloom_mania.generator_density import ExecutionLedger
    from keyloom_mania.main_style import STYLE_NAMES, evaluate_bars
    from keyloom_mania.patterns import analyze_file
    from keyloom_mania.timing import read_chart
    from keyloom_mania.catalog import verify_shared_resources
    verify_shared_resources()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    with np.load(args.features, allow_pickle=False) as features:
        acoustic, music = features['acoustic'], features['music']
    rhythm = np.load(args.rhythm, allow_pickle=False)
    settings = dict(bpm=args.bpm, first_beat_ms=args.offset, target_star=args.star,
        main_style='Jumpstream', secondary_styles=['Stream', 'Rolls'], model_id='music-base',
        arrangement_model_id='standard', density_model_id='contrast', budget_id='legacy',
        coordination_id='coordinated', title='Development audio 01', artist='Not redistributed')
    variants = ('legacy', 'hierarchy-only', 'hierarchy-shaped')
    protocol = {'scope': 'One previously used development song; training overlap and redistribution rights unresolved. No audio or source-song chart is published.',
        'settings': settings, 'seed': args.seed, 'decoder_rng_policy':'Checkpoint-specific decoder seeds remain unchanged; see effective policy identity.',
        'variants': list(variants), 'device': 'cuda',
        'source_sha256': args.source_sha256, 'feature_cache_sha256': sha(args.features), 'rhythm_cache_sha256': sha(args.rhythm),
        'input_cache_policy': 'Frozen original acoustic, MusicFM and rhythm frames reused identically; audio decoded for duration, encoder not rerun.',
        'source_sha256_by_file': {f: sha(ROOT/f) for f in ['scripts/review_real_song.py', 'keyloom_mania/studio/pipeline.py', 'keyloom_mania/hierarchical_planner.py', 'keyloom_mania/generator_density.py']},
        'catalog_sha256': sha(ROOT/'keyloom_mania/resources/catalog.json'),
        'style_evaluator': 'Native public OMTK intervals and evaluate_bars; no archived grid-stream overlay. Historical overlay percentages are not directly comparable.',
        'versions': {'torch': torch.__version__, 'numpy': np.__version__},
        'primary_comparison': 'Hierarchy-shaped versus hierarchy-only isolates density shaping conditional on hierarchy, with final roles permitted to respond to density.',
        'secondary_comparison': 'Hierarchy-only versus legacy changes structural segmentation and roles.',
        'failure_policy': 'Retain every attempted variant, error and output. Never rerun with altered settings to replace an adverse result.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    results = {}
    ledgers = []
    class CapturingLedger(ExecutionLedger):
        def __init__(self):
            super().__init__()
            ledgers.append(self)
    real_shape = pipeline.shape_hierarchical_density
    def unshaped(density, structure):
        _, report = real_shape(density, structure)
        report.update(multipliers=[1.0]*len(density), shaped_note_budget_sum=report['original_note_budget_sum'],
                      shaped_row_budget_sum=report['original_row_budget_sum'], changed_bars=0,
                      shaped_beat_density=np.asarray(density).tolist(), research_ablation='hierarchy_only_identity_density')
        return np.asarray(density).copy(), report
    for variant in variants:
        folder = args.output/variant
        folder.mkdir()
        source = folder/('source'+args.audio.suffix)
        shutil.copyfile(args.audio, source)
        requested = {**settings, 'structure_id': 'legacy' if variant=='legacy' else 'hierarchical'}
        bars = []
        def progress(stage, fraction, message, **extra):
            if 'bars' in extra:
                bars[:] = extra['bars']
            if stage != 'generation' or int(fraction*100)%10 == 0:
                print(json.dumps({'variant': variant, 'stage': stage, 'progress': round(fraction, 3)}), flush=True)
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        started = time.perf_counter()
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(pipeline, 'logmel_features', return_value=acoustic))
                stack.enter_context(patch('keyloom_mania.local_rhythm.rhythm_descriptors', return_value=rhythm))
                stack.enter_context(patch('keyloom_mania.music_features.build_encoder', return_value=(None, {})))
                stack.enter_context(patch('keyloom_mania.music_features.extract_embeddings', return_value=(music, {'audio_seconds_per_inference_second': 0.0})))
                stack.enter_context(patch.object(pipeline, 'ExecutionLedger', CapturingLedger))
                if variant == 'hierarchy-only':
                    stack.enter_context(patch.object(pipeline, 'shape_hierarchical_density', side_effect=unshaped))
                manifest = pipeline._generate_song(source, requested, progress, device='cuda')
            packaged_chart(folder,manifest['package_sha256'])
            (folder/'committed-ledger.json').write_text(json.dumps(sorted([t,*v] for t,v in ledgers[-1].events.items())), encoding='utf-8')
            bars = read(folder/'bar-density.json')['bars'] if not bars else bars
            clock, notes = read_chart(folder/'generated.osu')
            analysis = analyze_file(folder/'generated.osu')
            starts = np.array([b['start_ms'] for b in bars])
            ends = np.array([b['end_ms'] for b in bars])
            style_ids = np.array([STYLE_NAMES.index(b['style']) if b['style'] else 0 for b in bars])
            complete = ends-starts >= 4*60000/args.bpm-1
            adherence = evaluate_bars(analysis, starts, ends, style_ids, np.array([b['quiet'] for b in bars]), complete)
            (folder/'bar-adherence.json').write_text(json.dumps(adherence, ensure_ascii=False, indent=2), encoding='utf-8')
            trace = read(folder/'density-execution.json')
            identity = {k:manifest.get(k) for k in ['generator_sha256','arrangement_sha256','density_sha256','inference_policy_sha256','generator_runtime_policy']}
            results[variant] = {'status':'complete','notes':manifest['notes'],'rows':manifest['rows'],'actual_star':manifest['actual_star'],
                'sections':manifest['hierarchy_sections'],'phrases':manifest['hierarchy_phrases'],
                'execution':trace['summary'],'density':read(folder/'bar-density.json')['summary'],
                'adherence':{k:adherence[k] for k in ['eligible_bars','passed_bars','bar_pass_rate','per_style']},
                'identity':identity,'chart_sha256':sha(folder/'generated.osu'), 'elapsed_seconds':time.perf_counter()-started}
        except Exception as error:
            results[variant] = {'status':'failed','error_type':type(error).__name__,'error':str(error),'elapsed_seconds':time.perf_counter()-started}
        (args.output/'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'variant':variant, 'result':results[variant]},ensure_ascii=False),flush=True)
    if any(v['status']!='complete' for v in results.values()):
        raise RuntimeError('One or more fixed-protocol arms failed; retained in results.json.')
    identities = [r['identity'] for r in results.values()]
    if not all(x==identities[0] for x in identities):
        raise ValueError('Generator/adapter identities or effective policy changed across arms.')
    intermediate_audit={}
    with np.load(args.output/'legacy/planner-inputs.npz') as base:
        for variant in variants[1:]:
            with np.load(args.output/variant/'planner-inputs.npz') as candidate:
                for key in ['music','acoustic','rhythm','base_density','unshaped_beat_density']:
                    np.testing.assert_array_equal(base[key],candidate[key])
                intermediate_audit[variant]={key:{'bitwise_equal':bool(np.array_equal(base[key],candidate[key])),
                    'max_absolute_difference':float(np.max(abs(base[key].astype(np.float64)-candidate[key].astype(np.float64))))}
                    for key in ['owner_scores','style_evidence']}
    matched = {}
    for a,b in [('legacy','hierarchy-only'),('hierarchy-only','hierarchy-shaped'),('legacy','hierarchy-shaped')]:
        aa,bb = read(args.output/a/'bar-adherence.json')['bars'],read(args.output/b/'bar-adherence.json')['bars']
        pairs=[(x,y) for x,y in zip(aa,bb) if x['eligible'] and y['eligible'] and x['planned_style']==y['planned_style']]
        matched[a+'__'+b]={'bars':len(pairs),'passed_a':sum(x['pass'] for x,y in pairs),'passed_b':sum(y['pass'] for x,y in pairs),
            'rate_a':float(np.mean([x['pass'] for x,y in pairs])) if pairs else None,
            'rate_b':float(np.mean([y['pass'] for x,y in pairs])) if pairs else None}
    (args.output/'paired-summary.json').write_text(json.dumps({'identical_frozen_inputs_and_models':True,
        'inferred_intermediate_audit':intermediate_audit,'matched_style_comparisons':matched,
        'boundary':'Descriptive comparison on one previously used development song and one seed. Bars are dependent, not independent experimental replicates.'},indent=2),encoding='utf-8')
    print('Completed all three fixed-protocol development arms.',flush=True)


if __name__=='__main__':
    main()
