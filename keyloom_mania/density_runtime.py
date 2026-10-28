"""Density planning with explicit base-planner compatibility."""
import numpy as np
from .catalog import catalog, asset_path, verify_asset
from .features import hash_file


def density_options():
    return [{"id":"fixed","label":"原密度 r1","available":True,"status":"baseline"}] + [
        {"id":item["id"],"label":item["label"],"revision":item["revision"],"available":asset_path(item["checkpoint"]).is_file(),"status":"research"} for item in catalog()["density"]]


def resolve_density_model(model_id="fixed", planner_path=None):
    if model_id == "fixed": return None
    option = next((item for item in catalog()["density"] if item["id"] == model_id), None)
    if option is None: raise ValueError("Unsupported density model.")
    path = verify_asset(option["checkpoint"])
    if planner_path is not None and hash_file(planner_path) != option["planner_sha256"]:
        raise ValueError("Density adapter does not match the base planner.")
    return {**option,"path":path,"sha256":catalog()["files"][option["checkpoint"]]["sha256"]}


def infer_density(option, planner_path, memory, base_density, segment_ids, quiet, beat_activity, device="cuda"):
    import torch
    from .section_density import DENSITY_SCHEMA, SectionDensityAdapter, density_context

    if hash_file(planner_path) != option["planner_sha256"]:
        raise ValueError("密度适配器与当前 Planner 不匹配。")
    receipt = torch.load(option["path"], map_location="cpu", weights_only=True)
    if receipt["schema_version"] != DENSITY_SCHEMA or receipt["frozen_planner_sha256"] != option["planner_sha256"]:
        raise ValueError("密度适配器来源记录不匹配。")
    adapter = SectionDensityAdapter(receipt["width"]).to(device).eval()
    adapter.load_state_dict(receipt["state_dict"], strict=True)
    section, global_memory = density_context(memory, segment_ids)
    norm = receipt["normalization"]
    values = {"memory": memory, "section_memory": section,
              "global_memory": np.repeat(global_memory[None], len(memory), axis=0),
              "base_density": base_density, "beat_activity": (beat_activity - norm["mean"]) / norm["std"]}
    inputs = {name: torch.as_tensor(value, dtype=torch.float32, device=device)[None] for name, value in values.items()}
    inputs["quiet"] = torch.as_tensor(quiet, dtype=torch.bool, device=device)[None]
    with torch.inference_mode():
        output = adapter(**inputs)
    return output["beat_density"][0].cpu().numpy()
