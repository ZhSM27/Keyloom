"""Small public generator registry with family-scoped revision numbers."""
import json
from dataclasses import dataclass
from pathlib import Path
from ..catalog import asset_path, catalog, verify_asset
from .config import DEFAULT_GENERATOR_MODEL_ID


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class GeneratorModelOption:
    model_id: str
    label: str
    description: str
    checkpoint: Path
    checkpoint_sha256: str
    revision: int
    status: str = "research"
    recommended: bool = False
    notice: str = "Research release; assess musical quality by listening and playtesting."

    @property
    def export_name(self):
        return f"{self.model_id}-r{self.revision}"


STUDIO_GENERATOR_MODELS = tuple(GeneratorModelOption(
    model_id=item["id"], label=item["label"], description=item["description"],
    checkpoint=asset_path(item["checkpoint"]),
    checkpoint_sha256=catalog()["files"][item["checkpoint"]]["sha256"],
    revision=item["revision"],
) for item in catalog()["generators"])
STUDIO_GENERATOR_MODEL_BY_ID = {item.model_id: item for item in STUDIO_GENERATOR_MODELS}
DEFAULT_GENERATOR_MODEL = STUDIO_GENERATOR_MODEL_BY_ID[DEFAULT_GENERATOR_MODEL_ID]
OFFLINE_MODEL_NAME = DEFAULT_GENERATOR_MODEL.export_name


def generator_model_validation_error(model, *, verify_integrity=False):
    item = next(item for item in catalog()["generators"] if item["id"] == model.model_id)
    try:
        if verify_integrity:
            verify_asset(item["checkpoint"])
        elif not model.checkpoint.is_file():
            return "请安装本版本的模型资源包。"
    except (OSError, ValueError) as error:
        return str(error)
    return None


def resolve_generator_model(model_id=None, *, verify_integrity=True):
    requested = DEFAULT_GENERATOR_MODEL_ID if model_id is None else model_id
    if isinstance(requested, str):
        requested = requested.removeprefix("generator/")
    if not isinstance(requested, str) or requested not in STUDIO_GENERATOR_MODEL_BY_ID:
        raise ValueError("请选择受支持的生成模型。")
    model = STUDIO_GENERATOR_MODEL_BY_ID[requested]
    problem = generator_model_validation_error(model, verify_integrity=verify_integrity)
    if problem:
        raise ValueError(problem)
    return model


def public_generator_models(*, verify_integrity=False):
    result = []
    for model in STUDIO_GENERATOR_MODELS:
        problem = generator_model_validation_error(model, verify_integrity=verify_integrity)
        result.append({"id": model.model_id, "model_key": f"generator/{model.model_id}",
            "revision": model.revision, "label": model.label, "description": model.description,
            "status": model.status, "status_label": "研究", "recommended": False,
            "notice": model.notice, "available": problem is None, "unavailable_reason": problem,
            "checkpoint": f"generator/{model.model_id}-r{model.revision}.pt",
            "checkpoint_sha256": model.checkpoint_sha256})
    return result
