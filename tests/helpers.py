from keyloom_mania.studio.registry import STUDIO_GENERATOR_MODEL_BY_ID

def portable_generator(model_id=None, **kwargs):
    name = (model_id or "music-base").removeprefix("generator/")
    if name not in STUDIO_GENERATOR_MODEL_BY_ID:
        raise ValueError("Unsupported generator")
    return STUDIO_GENERATOR_MODEL_BY_ID[name]
