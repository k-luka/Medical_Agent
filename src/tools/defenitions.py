from pathlib import Path
from typing import Optional

from src.tools.inference import get_model
from .analysis import MedicalAnalyzer

def segment_spleen_ct(image_path: str):
    """
    Segmetns spleen from CT scan
    """
    model = get_model("spleen")
    if not model:
        return "Tool Unavailable: Spleen model not loaded."
    return model.predict(image_path)

# def segment_brain_mri(image_path: str):
#     """
#     Segments brain tumor from MRI scan
#     """
#     model = get_model("brain")
#     if not model:
#         return "Tool Unavailable: Brain model not loaded."
#     return model.predict(image_path)


def segment_multi_organ_ct(image_path: str):
    """
    Segments 13 abdominal organs (spleen, kidney, liver, stomach, etc.) at once from a CT scan.
    """
    model = get_model("multi_organ")
    if not model:
        return "Tool Unavailable: Multi-organ model not loaded."
    return model.predict(image_path)

def lung_nodule_ct_detection(image_path: str):
    """
    Detects and localizes suspicious pulmonary nodules in Chest CT scans, providing bounding boxes and confidence scores.
    """
    model = get_model("lung_nodule_ct_detection")
    
    if not model:
        return "Tool Unavailable: Lung detection model not loaded."
        
    return model.predict(image_path)

analyzer: Optional[MedicalAnalyzer] = None


def init_analysis(sandbox_path: Path) -> None:
    """
    Configure the analyzer with a sandbox path (called from Hydra entrypoint).
    """
    global analyzer
    analyzer = MedicalAnalyzer(Path(sandbox_path))

def inspect_segmentation_tool(mask_filename: str, ct_filename: str):
    """
    Analyzes a segmentation mask.
    ARGS:
        mask_filename: The name of the mask file (e.g., 'segmentation.nii.gz')
        ct_filename: The name of the original CT file (e.g., 'scan.nii.gz')
    """
    if analyzer is None:
        raise RuntimeError("Analyzer not initialized. Call init_analysis() with a sandbox path.")

    metrics = analyzer.analyze_mask(mask_filename, ct_filename)
    image_name = analyzer.save_center_slice(mask_filename, ct_filename)
    
    return {
        "status": "success",
        "organ_metrics": metrics,
        "visual_file": image_name # The agent can now use this filename to display the image
    }


def view_saved_slice(image_filename: str):
    """
    Returns a base64-encoded PNG of a saved slice inside the sandbox.
    """
    if analyzer is None:
        raise RuntimeError("Analyzer not initialized. Call init_analysis() with a sandbox path.")
    encoded = analyzer.read_slice_png(image_filename)
    return {"status": "success", "image_base64": encoded}
