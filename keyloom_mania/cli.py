"""Portable command line entry point."""
import argparse
import json
import shutil
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
    generate.add_argument("--structure", choices=("legacy", "hierarchical"), default="legacy")
    generate.add_argument("--budget", choices=("legacy", "count-reference"), default="legacy")
    generate.add_argument("--coordination", default="legacy")
    generate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
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
        else:
            from .studio.config import OUTPUT_ROOT
            from .studio.pipeline import _generate_song
            from .studio.contracts import validate_settings
            settings = {"bpm": args.bpm, "first_beat_ms": args.offset, "target_star": args.star,
                "main_style": args.style, "secondary_styles": args.secondary, "model_id": args.model,
                "arrangement_model_id": args.role, "density_model_id": args.density,
                "structure_id": args.structure, "budget_id": args.budget, "coordination_id": args.coordination,
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
