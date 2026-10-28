"""Explicit integrity and native-load checks without a local training checkout."""
from .catalog import catalog, verify_asset


def check(*, load_models=False, device="cpu"):
    result = {"ok": True, "assets": {}, "models": {}, "device": device}
    for name in catalog()["files"]:
        try:
            verify_asset(name)
            result["assets"][name] = "ok"
        except (OSError, ValueError) as error:
            result["assets"][name] = str(error)
            result["ok"] = False
    from .audio_assets import DEFAULT_ASSET_ROOT, asset_specification, check_asset
    for name, specification in asset_specification().items():
        try:
            check_asset(DEFAULT_ASSET_ROOT / name, specification)
            result["assets"]["audio/musicfm/" + name] = "ok"
        except (OSError, ValueError):
            result["assets"]["audio/musicfm/" + name] = "missing or invalid; run assets fetch-audio"
            result["ok"] = False
    if load_models:
        from .studio.loader import _generated_plan_model
        for model in catalog()["generators"]:
            try:
                _generated_plan_model(verify_asset(model["checkpoint"]), device=device)
                result["models"][model["id"]] = "native load ok"
            except Exception as error:
                result["models"][model["id"]] = f"{type(error).__name__}: {error}"
                result["ok"] = False
    return result
