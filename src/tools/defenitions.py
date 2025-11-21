from src.tools.inference import get_model

def segment_spleen_ct(image_path: str):
    """
    Segmetns spleen from CT scan
    """
    model = get_model("spleen")
    if not model:
        return "Tool Unavailable: Spleen model not loaded."
    return model.predict(image_path)

def segment_brain_mri(image_path: str):
    """
    Segments brain tumor from MRI scan
    """
    model = get_model("brain")
    if not model:
        return "Tool Unavailable: Brain model not loaded."
    return model.predict(image_path)