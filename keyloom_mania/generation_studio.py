from __future__ import annotations

import argparse
import email.parser
import email.policy
import json
import os
import re
import socket
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .features import AcousticConfig, decode_waveform, hash_file
from .catalog import CATALOG_PATH
from .role_style_plan import ROLE_STYLE_VERSION, active_style_plan, validate_style_selection
from .density_coordination import COORDINATION_OPTIONS, COORDINATION_VERSION, validate_coordination
from .density_runtime import density_options, resolve_density_model
from .hierarchical_planner import STRUCTURE_OPTIONS, HIERARCHY_VERSION, validate_structure_id
from .generator_density import BUDGET_OPTIONS, BUDGET_VERSION, DIAGNOSTIC_VERSION, validate_budget
from .role_planner_runtime import arrangement_options, resolve_arrangement_model
from .segment_labels import LABELS

from .studio.config import (WEB_ROOT, PLANNER_CHECKPOINT, PLAN_ARTIFACT, PLAN_ANALYSIS,
    PLANNER_NORMALIZATION, GENERATOR_NORMALIZATION, STYLE_STATISTICS, DEFAULT_MUSICFM_ASSETS,
    MUSICFM_REQUIRED_ASSETS, OUTPUT_ROOT, MAX_UPLOAD_BYTES, ALLOWED_EXTENSIONS,
    BAR_BEATS, CANDIDATES_PER_BEAT, STUDIO_RUNTIME_VERSION, STUDIO_PORT, STUDIO_URL)
from .studio.registry import (GeneratorModelOption, STUDIO_GENERATOR_MODELS, STUDIO_GENERATOR_MODEL_BY_ID, DEFAULT_GENERATOR_MODEL_ID, DEFAULT_GENERATOR_MODEL, OFFLINE_MODEL_NAME, generator_model_validation_error, resolve_generator_model, public_generator_models)
from .studio.loader import _generated_plan_model
from .studio.pipeline import (_generate_song, _normalize_features, _timing_inputs, _planner_phase, pool_feature_rows, sample_clamped)
from .tempo_detection import estimate_tempo
from .studio.contracts import validate_settings, named_options, option_id

class StudioApp:
    def __init__(self, output_root: Path, port: int):
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.port = port
        self.jobs: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.generation_lock = threading.Lock()

    def update(self, job_id: str, **fields):
        with self.lock:
            job = self.jobs[job_id]
            job.update(fields, updated_at=time.time())
            persisted = {key: value for key, value in job.items() if key != "source_path"}
            (Path(job["directory"]) / "status.json").write_text(json.dumps(persisted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def upload(self, filename: str, content: bytes) -> dict:
        name = Path(filename.replace("\\", "/")).name
        suffix = Path(name).suffix.lower()
        if not name or suffix not in ALLOWED_EXTENSIONS or len(content) < 4096:
            raise ValueError("请选择一首有效的 MP3、OGG、WAV、FLAC 或 M4A 音频。")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("单个音频文件最多 180 MiB。")
        job_id = uuid.uuid4().hex
        directory = self.output_root / job_id
        directory.mkdir(parents=True)
        source_path = directory / ("source" + suffix)
        source_path.write_bytes(content)
        now = time.time()
        job = {
            "id": job_id,
            "filename": name,
            "directory": str(directory),
            "source_path": str(source_path),
            "status": "analyzing",
            "stage": "analyzing",
            "progress": 0.03,
            "message": "读取音频并估算拍速",
            "bpm": None,
            "first_beat_ms": 0.0,
            "created_at": now,
            "updated_at": now,
        }
        with self.lock:
            self.jobs[job_id] = job
        threading.Thread(target=self._analyze, args=(job_id,), daemon=True, name=f"audio-analysis-{job_id[:8]}").start()
        return self.public_job(job_id)

    def _analyze(self, job_id: str):
        source_path = Path(self.jobs[job_id]["source_path"])
        try:
            waveform, metadata = decode_waveform(source_path, AcousticConfig())
            from .beat_timing import detect_timing
            try:
                tempo = detect_timing(waveform)
                (source_path.parent / 'timing.json').write_text(json.dumps(tempo, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception as detection_error:
                tempo = {'bpm': None, 'offset_ms': None, 'confidence': 0., 'status': 'unavailable',
                         'method': 'beat-this-final0', 'alternatives': [],
                         'note': f'自动检测未完成：{detection_error}；仍可手动输入 BPM 和 offset 生成。'}
            tags = {}
            try:
                from mutagen import File
                tags = File(str(source_path), easy=True) or {}
                title = str((tags.get("title") or [Path(self.jobs[job_id]["filename"]).stem])[0])
                artist = str((tags.get("artist") or ["Unknown artist"])[0])
            except Exception:
                title, artist = Path(self.jobs[job_id]["filename"]).stem, "Unknown artist"
            self.update(
                job_id,
                status="ready",
                stage="ready",
                progress=0.08,
                message=tempo["note"],
                duration_ms=round(float(metadata["decoded_duration_seconds"]) * 1000, 1),
                detected_bpm=tempo["bpm"],
                detected_offset_ms=tempo["offset_ms"],
                confidence=tempo["confidence"],
                tempo_alternatives=tempo["alternatives"],
                tempo_status=tempo["status"],
                tempo_method=tempo["method"],
                tempo_note=tempo['note'],
                grid_error_p90_ms=tempo.get('grid_error_p90_ms'),
                beat_count=tempo.get('beat_count', 0),
                downbeat_count=tempo.get('downbeat_count', 0),
                title=title,
                artist=artist,
                decoded_audio={"sample_rate": metadata["native_sample_rate"], "channels": metadata["native_channels"]},
            )
            del waveform
        except Exception as error:
            self.update(job_id, status="failed", stage="failed", progress=0, message=f"音频校验失败：{error}")

    def generate(self, job_id: str, settings: dict):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Unknown audio upload session.")
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job["status"] not in ("ready", "complete", "failed"):
                raise ValueError("Please wait until audio analysis finishes before generating.")
            if self.generation_lock.locked():
                raise RuntimeError("Another chart is being generated. Please wait for it to finish.")
            source_path = Path(job["source_path"])
            generator_model = resolve_generator_model(settings.get("model_id"))
            density_model_id = option_id("planner/density", settings.get("density_model_id", "fixed"))
            resolve_density_model(density_model_id, PLANNER_CHECKPOINT)
            arrangement_model_id = option_id("planner/role", settings.get("arrangement_model_id", "rule"))
            resolve_arrangement_model(arrangement_model_id)
            coordination_id=validate_coordination(settings.get('coordination_id','legacy'))
            structure_id=validate_structure_id(settings.get('structure_id','legacy'))
            budget_id=validate_budget(settings.get('budget_id','legacy'), for_studio=True, generator_id=generator_model.model_id)
            settings = {
                "bpm": float(settings.get("bpm", job.get("detected_bpm") or 120)),
                "first_beat_ms": float(settings.get("first_beat_ms", job.get("detected_offset_ms") or 0)),
                "target_star": float(settings.get("target_star", 5)),
                "main_style": str(settings.get("main_style", "Jumpstream")),
                "secondary_styles": settings.get("secondary_styles", settings.get("secondary_style", [])),
                "density_model_id": density_model_id,
                "arrangement_model_id": arrangement_model_id,
                "coordination_id":coordination_id,
                "structure_id":structure_id,
                "budget_id":budget_id,
                "title": str(settings.get("title", job.get("title", Path(job["filename"]).stem))),
                "artist": str(settings.get("artist", job.get("artist", "Unknown artist"))),
                "model_id": generator_model.model_id,
            }
            validate_settings(settings)
            _, secondary_labels = validate_style_selection(settings["main_style"], settings["secondary_styles"])
            settings["secondary_styles"] = list(secondary_labels)
            if budget_id == 'count-reference' and settings['main_style'] != 'Jumpstream':
                raise ValueError('音乐行数与押数协调目前仅开放 Jumpstream 主型；其他主型请保留原密度条件。')
            if source_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                raise ValueError("The uploaded audio file is no longer available.")
            settings["bpm"] = round(settings["bpm"], 3)
            if not self.generation_lock.acquire(blocking=False):
                raise RuntimeError("Another chart is being generated.")
            try:
                self.update(
                    job_id,
                    status="queued",
                    stage="queued",
                    progress=0.1,
                    message=f"已加入本地 GPU 生成队列 · {generator_model.label}",
                    settings=settings,
                    manifest=None,
                    bars=None,
                    filename=None,
                )
                threading.Thread(target=self._run_generation, args=(job_id, source_path, settings), daemon=True, name=f"chart-generation-{job_id[:8]}").start()
            except Exception:
                self.generation_lock.release()
                self.update(job_id, status="failed", stage="failed", message="Could not start generation.")
                raise

    def _run_generation(self, job_id: str, audio_path: Path, settings: dict):
        try:
            self.update(job_id, status="running", stage="decode", progress=0.04)

            def progress(stage, value, message, **extra):
                self.update(job_id, stage=stage, progress=round(float(value), 3), message=message, **extra)

            _generate_song(audio_path, settings, progress)
            self.update(job_id, status="complete", stage="complete", progress=1.0)
        except Exception as error:
            self.update(job_id, status="failed", stage="failed", progress=0, message=f"生成未完成：{type(error).__name__}: {error}")
        finally:
            self.generation_lock.release()

    def public_job(self, job_id: str) -> dict:
        with self.lock:
            job = self.jobs[job_id]
            return {key: value for key, value in job.items() if key not in {"source_path", "directory", "decoded_audio"}}

    @staticmethod
    def health() -> dict:
        required_assets = (
            PLANNER_NORMALIZATION,
            PLANNER_CHECKPOINT,
            PLAN_ARTIFACT,
            PLAN_ANALYSIS,
            STYLE_STATISTICS,
            GENERATOR_NORMALIZATION,
            *MUSICFM_REQUIRED_ASSETS,
        )
        missing_assets = [value.name for value in required_assets if not value.is_file()]
        generator_options = public_generator_models(verify_integrity=False)
        available_generators = [model for model in generator_options if model["available"]]
        default_generator = next(
            (model for model in generator_options if model["id"] == DEFAULT_GENERATOR_MODEL_ID and model["available"]),
            None,
        )
        try:
            import torch
            from .playstyle_control import DECODER_VERSION
            has_gpu = bool(torch.cuda.is_available())
            device = torch.cuda.get_device_name(0) if has_gpu else "CUDA unavailable"
            runtime_error = None if has_gpu else "请安装匹配的 CUDA 版 PyTorch，或使用离线 CPU 生成命令。"
        except Exception as error:
            has_gpu = False
            device = "CUDA runtime unavailable"
            runtime_error = f"本地 CUDA 运行环境不可用：{type(error).__name__}。请从项目根目录双击 studio.bat。"
            DECODER_VERSION = None
        if not available_generators:
            runtime_error = runtime_error or "工作台没有找到可用的生成器检查点。"
        models_available = not missing_assets and has_gpu and bool(default_generator)
        status = (
            "ok" if models_available
            else "models_missing" if missing_assets or not available_generators
            else "runtime_unavailable"
        )
        return {
            "status": status,
            "models_available": models_available,
            "generator": default_generator["label"] if default_generator else None,
            "generator_checkpoint": default_generator["checkpoint"] if default_generator else None,
            "default_generator_id": default_generator["id"] if default_generator else None,
            "generator_options": generator_options,
            "decoder_version": DECODER_VERSION,
            "generation_runtime_version": STUDIO_RUNTIME_VERSION,
            "model_catalog_sha256": hash_file(CATALOG_PATH),
            "planner_policy_version": ROLE_STYLE_VERSION,
            "density_options": named_options("planner/density", density_options()),
            "default_density_model_id": "fixed",
            "arrangement_options": named_options("planner/role", arrangement_options()),
            "default_arrangement_model_id": "rule",
            "coordination_options":list(COORDINATION_OPTIONS),"default_coordination_id":"legacy",
            "coordination_version":COORDINATION_VERSION,
            "structure_options":list(STRUCTURE_OPTIONS), "default_structure_id":"legacy", "hierarchy_version":HIERARCHY_VERSION,
            "budget_options":list(BUDGET_OPTIONS), "default_budget_id":"legacy",
            "budget_version":BUDGET_VERSION, "execution_diagnostic_version":DIAGNOSTIC_VERSION,
            "planner_integration_version":"planner-suite-r1",
            "generator_source_runtime":"installed-package",
            "studio_port": STUDIO_PORT,
            "pid": os.getpid(),
            "pressure_windows_beats":[2,4],
            "pressure_semantics_notice":"Released models use finite 2/4-beat hand observations.",
            "missing": missing_assets,
            "runtime_error": runtime_error,
            "device": device,
            "formats": sorted(ALLOWED_EXTENSIONS),
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "styles": list(LABELS),
            "listen": "127.0.0.1",
        }


class StudioHTTPServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR can allow two live listeners on the same address.
    allow_reuse_address = False

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "Keyloom-Studio/1.0"

    def __init__(self, *args, app: StudioApp, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    def log_message(self, _format, *args):
        return

    def _send(self, payload: bytes, content_type: str, status=200, extra_headers: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, value: dict, status=200):
        self._send(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/health":
            with self.app.lock:
                active_jobs = sum(job["status"] in ("queued", "running", "analyzing", "decoding") for job in self.app.jobs.values())
            self._json({**self.app.health(), "active_jobs": active_jobs})
            return
        match = re.fullmatch(r"/api/session/([0-9a-f]{32})", path)
        if match:
            if match.group(1) not in self.app.jobs:
                self._json({"error": "Unknown audio upload session."}, 404)
                return
            self._json(self.app.public_job(match.group(1)))
            return
        match = re.fullmatch(r"/api/download/([0-9a-f]{32})", path)
        if match:
            job = self.app.jobs.get(match.group(1))
            if not job or job.get("status") != "complete" or not job.get("filename"):
                self._json({"error": "The chart package is not ready."}, 404)
                return
            package = Path(job["directory"]) / job["filename"]
            if not package.is_file() or package.parent.resolve() != self.app.output_root.joinpath(match.group(1)).resolve():
                self._json({"error": "The generated `.osz` is unavailable."}, 404)
                return
            self._send(package.read_bytes(), "application/octet-stream", extra_headers={"Content-Disposition": f"attachment; filename*=UTF-8''{__import__('urllib.parse', fromlist=['quote']).quote(package.name)}"})
            return
        if path in ("/", "/app.js", "/style.css"):
            filename = "index.html" if path == "/" else path[1:]
            content_type = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8"}[filename]
            try:
                self._send((WEB_ROOT / filename).read_bytes(), content_type)
            except OSError:
                self._json({"error": "User interface assets are missing."}, 404)
            return
        self._json({"error": "Not found."}, 404)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and origin != STUDIO_URL.rstrip("/"):
            self._json({"error": "Cross-origin requests are not allowed."}, 403)
            return
        path = urlsplit(self.path).path
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header or "-1")
        except ValueError:
            length = -1
        if length < 0 or length > MAX_UPLOAD_BYTES + 64 * 1024:
            self._json({"error": "Request body is missing or exceeds the 180 MiB limit."}, 413)
            return
        body = self.rfile.read(length)
        try:
            if path == "/api/upload":
                content_type = self.headers.get("Content-Type", "")
                if not content_type.lower().startswith("multipart/form-data"):
                    raise ValueError("Upload an audio file using multipart/form-data.")
                envelope = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + body
                message = email.parser.BytesParser(policy=email.policy.default).parsebytes(envelope)
                file_part = next((part for part in message.iter_parts() if part.get_content_disposition() == "form-data" and part.get_param("name", header="content-disposition") == "audio"), None)
                if file_part is None:
                    raise ValueError("Choose one audio file before uploading.")
                filename = file_part.get_filename() or "song.mp3"
                payload = file_part.get_payload(decode=True) or b""
                self._json({"session": self.app.upload(filename, payload)}, 201)
                return
            match = re.fullmatch(r"/api/generate/([0-9a-f]{32})", path)
            analysis_match = re.fullmatch(r"/api/analyze/([0-9a-f]{32})", path)
            if analysis_match:
                job_id = analysis_match.group(1)
                with self.app.lock:
                    job = self.app.jobs.get(job_id)
                    if not job:
                        self._json({"error": "请先上传歌曲。"}, 404)
                        return
                    if job['status'] in ('running', 'queued', 'analyzing'):
                        raise RuntimeError("当前任务正在处理，请稍后再测量。")
                    self.app.update(job_id, status='analyzing', stage='analyzing', message='自动测量 BPM 和 offset…')
                threading.Thread(target=self.app._analyze, args=(job_id,), daemon=True).start()
                self._json(self.app.public_job(job_id), 202)
                return
            if match:
                settings = json.loads(body.decode("utf-8"))
                if not isinstance(settings, dict):
                    raise ValueError("Generation settings must be a JSON object.")
                self.app.generate(match.group(1), settings)
                self._json(self.app.public_job(match.group(1)), 202)
                return
            self._json({"error": "Not found."}, 404)
        except RuntimeError as error:
            self._json({"error": str(error)}, 409)
        except (ValueError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self._json({"error": f"Request failed: {type(error).__name__}: {error}"}, 500)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local interactive osu!mania 4K song and style generation studio.")
    parser.add_argument("--port", type=int, default=STUDIO_PORT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--open", action="store_true", help="Open the local studio in the default browser.")
    args = parser.parse_args(argv)
    if args.port != STUDIO_PORT:
        parser.error(f"The unified workbench uses port {STUDIO_PORT}.")
    app = StudioApp(args.output, args.port)
    try:
        server = StudioHTTPServer(("127.0.0.1", args.port), lambda *values, **kwargs: StudioHandler(*values, app=app, **kwargs))
    except OSError as error:
        from urllib.request import urlopen
        try:
            with urlopen(STUDIO_URL + "api/health", timeout=2) as response:
                health = json.load(response)
            if (health.get("generation_runtime_version") != STUDIO_RUNTIME_VERSION
                    or health.get("model_catalog_sha256") != hash_file(CATALOG_PATH)):
                raise ValueError("Different service occupies the workbench port.")
        except Exception:
            raise RuntimeError(f"Port {STUDIO_PORT} is occupied by another service; stop it before launching.") from error
        print("Workbench already running: " + STUDIO_URL)
        if args.open:
            webbrowser.open(STUDIO_URL)
        return
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Keyloom generation studio: {url}", flush=True)
    print(f"Generated packages: {app.output_root}", flush=True)
    if args.open:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("Stopping generation studio.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
