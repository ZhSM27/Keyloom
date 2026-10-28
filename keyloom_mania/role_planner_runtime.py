"""Conditional role planning from the public release catalog."""
import numpy as np
from .catalog import catalog, asset_path, verify_asset
from .music_structure import music_structure
from .segment_labels import LABELS


def arrangement_options():
    return [{"id":"rule", "label":"规则编排 r1", "available":True,"status":"baseline"}] + [
        {"id":item["id"],"label":item["label"],"revision":item["revision"],"parameter_count":item["parameter_count"],
         "available":asset_path(item["checkpoint"]).is_file(),"status":"research"} for item in catalog()["roles"]]


def resolve_arrangement_model(model_id="rule"):
    if model_id == "rule": return None
    option = next((item for item in catalog()["roles"] if item["id"] == model_id), None)
    if option is None: raise ValueError("Unsupported role planner.")
    return {**option,"path":verify_asset(option["checkpoint"]),"sha256":catalog()["files"][option["checkpoint"]]["sha256"]}


def infer_arrangement(option, music, acoustic, rhythm, phase, star, primary_label, secondary_labels, quiet, device="cuda"):
    import torch
    from .conditional_role_planner import ROLE_MODEL_SCHEMA, ConditionalRoleConfig, ConditionalRolePlanner

    receipt = torch.load(option["path"], map_location="cpu", weights_only=True)
    configuration = dict(receipt["config"])
    if configuration.pop("schema_version") != ROLE_MODEL_SCHEMA:
        raise ValueError("Unsupported conditional role Planner architecture.")
    # Model construction must not shift the downstream generator's random stream.
    with torch.random.fork_rng(devices=[]):
        model = ConditionalRolePlanner(ConditionalRoleConfig(**configuration)).to(device).eval()
    model.load_state_dict(receipt["state_dict"], strict=True)
    structure = music_structure(np.ones(len(music)), np.ones(len(music)), np.zeros((len(music), 15)), quiet,
                                music=music, acoustic=acoustic, rhythm=rhythm)
    audio = {}
    for name, values in (("music", music), ("acoustic", acoustic), ("rhythm", rhythm)):
        norm = receipt["normalization"][name]
        values = (np.asarray(values) - norm["mean"]) / norm["std"]
        audio[name] = torch.as_tensor(np.zeros_like(values) if receipt["arm"] == "no-audio" else values, dtype=torch.float32, device=device)[None]
    secondary = np.zeros((1, 15), dtype=np.float32)
    secondary[0, [LABELS.index(label) for label in secondary_labels]] = 1
    inputs = {**audio, "phase": torch.as_tensor(phase, dtype=torch.float32, device=device)[None],
              "star": torch.tensor([(star - 2) / 7], dtype=torch.float32, device=device),
              "primary": torch.tensor([LABELS.index(primary_label)], device=device),
              "secondary": torch.as_tensor(secondary, device=device),
              "segment_ids": torch.as_tensor(np.zeros(len(music), dtype=np.int32) if receipt["arm"] == "no-audio" else structure["segment_ids"], device=device)[None],
              "padding_mask": torch.zeros((1, len(music)), dtype=torch.bool, device=device)}
    with torch.inference_mode():
        output = model(**inputs)
    return output["owner_logits"][0].float().softmax(-1).cpu().numpy(), output["style_logits"][0].float().softmax(-1).cpu().numpy()
