"""Portable command line entry point."""
import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="keyloom")
    commands = parser.add_subparsers(dest="command", required=True)
    studio = commands.add_parser("studio", help="Open the single local workbench on port 8788")
    studio.add_argument("--no-open", action="store_true")
    commands.add_parser("models", help="List the released generator families")
    doctor = commands.add_parser("doctor", help="Check assets and runtime")
    doctor.add_argument("--load-models", action="store_true")
    doctor.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    assets = commands.add_parser("assets").add_subparsers(dest="action", required=True)
    assets.add_parser("install").add_argument("bundle", type=Path)
    assets.add_parser("fetch-audio").add_argument("--beats", action="store_true", help="Also download optional Beat This! weights")
    generate = commands.add_parser("generate", help="Generate from explicit timing, including on CPU")
    generate.add_argument("audio", type=Path)
    generate.add_argument("--bpm", type=float, required=True)
    generate.add_argument("--offset", type=float, default=0)
    generate.add_argument("--star", type=float, default=5)
    generate.add_argument("--style", default="Jumpstream")
    generate.add_argument("--secondary", action="append", default=[])
    generate.add_argument("--model", default="music-base")
    generate.add_argument("--role", choices=("rule", "standard", "balanced"), default="rule")
    generate.add_argument("--density", choices=("fixed", "contrast"), default="fixed")
    generate.add_argument("--structure", choices=("legacy", "hierarchical", "peak-anchored"), default="legacy")
    generate.add_argument("--star-condition", choices=("global", "section"), default="global",
                          help="section: lower the generator star in non-peak sections (requires --structure peak-anchored)")
    generate.add_argument("--budget", choices=("legacy", "count-reference"), default="legacy")
    generate.add_argument("--coordination", default="legacy")
    generate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate = commands.add_parser("evaluate", help="Score charts against human reference charts").add_subparsers(dest="action", required=True)
    reference = evaluate.add_parser("reference", help="Build a human reference profile from chart folders")
    reference.add_argument("roots", type=Path, nargs="*", help="osu! Songs folders, the osu!lazer files folder or chart files")
    reference.add_argument("--manifest", type=Path, action="append", default=[], help="corpus manifest.jsonl pairing charts with audio")
    reference.add_argument("--split", action="append", default=None, help="manifest splits to include (default: train only)")
    reference.add_argument("--output", type=Path)
    reference.add_argument("--max-hold-fraction", type=float, default=.1)
    reference.add_argument("--min-notes", type=int, default=200)
    reference.add_argument("--no-audio", action="store_true")
    reference.add_argument("--workers", type=int)
    chart = evaluate.add_parser("chart", help="Evaluate .osz packages, .osu charts or generation folders")
    chart.add_argument("charts", type=Path, nargs="+")
    chart.add_argument("--reference", type=Path)
    chart.add_argument("--neighbors", type=int, default=40)
    chart.add_argument("--no-audio", action="store_true")
    chart.add_argument("--json", type=Path, help="Also write the complete reports as JSON")
    chart.add_argument("--markdown", type=Path, help="Also write the printed report as UTF-8 Markdown")
    args = parser.parse_args(argv)
    try:
        if args.command == "studio":
            from .generation_studio import main as serve
            serve([] if args.no_open else ["--open"])
        elif args.command == "models":
            from .studio.registry import public_generator_models
            print(json.dumps(public_generator_models(), ensure_ascii=False, indent=2))
        elif args.command == "assets":
            from .assets import install_bundle, fetch_audio
            if args.action == "install":
                print(f"Installed {install_bundle(args.bundle)} verified resources.")
            else:
                fetch_audio(args.beats)
        elif args.command == "doctor":
            from .doctor import check
            result = check(load_models=args.load_models, device=args.device)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ok"] else 1
        elif args.command == "evaluate":
            return _evaluate(args)
        else:
            from .studio.config import OUTPUT_ROOT
            from .studio.pipeline import _generate_song
            from .studio.contracts import validate_settings
            settings = {"bpm": args.bpm, "first_beat_ms": args.offset, "target_star": args.star,
                "main_style": args.style, "secondary_styles": args.secondary, "model_id": args.model,
                "arrangement_model_id": args.role, "density_model_id": args.density,
                "structure_id": args.structure, "star_condition_id": args.star_condition, "budget_id": args.budget, "coordination_id": args.coordination,
                "title": args.audio.stem, "artist": "Unknown artist"}
            validate_settings(settings)
            if not args.audio.is_file():
                raise ValueError("Audio file does not exist.")
            directory = OUTPUT_ROOT / uuid.uuid4().hex
            directory.mkdir(parents=True)
            audio = directory / ("source" + args.audio.suffix.lower())
            shutil.copyfile(args.audio, audio)
            result = _generate_song(audio, settings, lambda stage, value, message, **extra: print(message, flush=True), device=args.device)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"Output: {directory}")
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


def _evaluate(args):
    from . import evaluation

    # Legacy console code pages cannot encode every report symbol; never fail on output.
    sys.stdout.reconfigure(errors="replace")
    if args.action == "reference":
        if not args.roots and not args.manifest:
            raise ValueError("Give chart folders or at least one --manifest.")
        profile = evaluation.build_reference(args.roots, args.output or evaluation.DEFAULT_REFERENCE,
            manifests=args.manifest, splits=tuple(args.split or ("train",)), maximum_hold_fraction=args.max_hold_fraction, minimum_notes=args.min_notes,
            with_audio=not args.no_audio, workers=args.workers)
        print(json.dumps(profile["summary"], ensure_ascii=False, indent=2))
        print(f"Reference: {args.output or evaluation.DEFAULT_REFERENCE}")
        return 0
    reference_path = args.reference or evaluation.DEFAULT_REFERENCE
    reference = evaluation.load_reference(reference_path) if args.reference or reference_path.is_file() else None
    if reference is None:
        print("No human reference profile; printing raw metrics only. Run `keyloom evaluate reference <folders>`.\n")
    reports, sections = [], []
    for path in args.charts:
        with evaluation.load_chart(path) as chart:
            reports.append(evaluation.evaluate_chart(chart, reference, with_audio=not args.no_audio, neighbor_count=args.neighbors))
        sections.append(evaluation.render_markdown(reports[-1]))
        print(sections[-1] + "\n", flush=True)
    if len(reports) > 1 and reference is not None:
        sections.append(evaluation.render_comparison(reports))
        print(sections[-1])
    if args.markdown:
        args.markdown.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if all(report["overall"]["gate_passed"] for report in reports) else 2
