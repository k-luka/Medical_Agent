from pathlib import Path
from typing import Any, Optional, cast
from nibabel.loadsave import load as nib_load
from src.tools.inference import get_model
from .analysis import MedicalAnalyzer
from ddgs import DDGS
import subprocess
import sys

# --- Globals ---
analyzer: Optional[MedicalAnalyzer] = None
SANDBOX_ROOT: Optional[Path] = None

# --- Initialization & Security ---
def init_paths(sandbox_path: Path) -> None:
    """
    Configure shared sandbox-aware helpers (called from entrypoints).
    """
    global analyzer, SANDBOX_ROOT
    SANDBOX_ROOT = Path(sandbox_path).resolve()
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    analyzer = MedicalAnalyzer(SANDBOX_ROOT)

def _resolve_in_sandbox(rel_path: str) -> Path:
    """
    Security Gatekeeper: Ensures path is strictly inside the sandbox.
    """
    if SANDBOX_ROOT is None:
        raise RuntimeError("Sandbox not initialized. Call init_paths() first.")
    
    # robust resolution
    try:
        full_path = (SANDBOX_ROOT / rel_path).resolve()
        full_path.relative_to(SANDBOX_ROOT)  # raises ValueError if outside
    except ValueError:
        raise PermissionError(f"Security Violation: Attempted access outside sandbox: {rel_path}")

    if not full_path.exists():
        raise FileNotFoundError(f"File not found in sandbox: {rel_path}")
    return full_path


# --- AI Tools ---

def segment_spleen_ct(image_path: str):
    """
    Segments spleen from CT scan
    """
    model = get_model("spleen")
    if not model:
        return "Tool Unavailable: Spleen model not loaded."
    try:
        resolved = _resolve_in_sandbox(image_path)
        return model.predict(str(resolved))
    except Exception as e:
        return f"Error: {str(e)}"

def segment_multi_organ_ct(image_path: str, organs: str = "all"):
    """
    Segments abdominal organs from a CT scan.
    
    ARGS:
        image_path: Filename of the CT scan.
        organs: A comma-separated list of organs to keep (e.g., "liver, spleen, pancreas"). 
                Defaults to "all" to keep everything.
                Valid options: spleen, right kidney, left kidney, gallbladder, esophagus, 
                liver, stomach, aorta, ivc, pancreas, right_adrenal, left_adrenal.
    """
    model = get_model("multi_organ")
    if not model:
        return "Tool Unavailable: Multi-organ model not loaded."
    
    try:
        resolved = _resolve_in_sandbox(image_path)
        
        # Process the input string into a list
        target_list = None
        if organs and organs.lower() != "all":
            # "liver, spleen" -> ["liver", "spleen"]
            target_list = [o.strip() for o in organs.split(",")]
            
        # Pass the list to the modified predict method
        return model.predict(str(resolved), target_organs=target_list)
        
    except Exception as e:
        return f"Error: {str(e)}"

# def lung_nodule_ct_detection(image_path: str):
#     """
#     Detects and localizes suspicious pulmonary nodules in Chest CT scans, providing bounding boxes and confidence scores.
#     """
#     model = get_model("lung_nodule_ct_detection")
#     if not model:
#         return "Tool Unavailable: Lung detection model not loaded."
#     try:
#         resolved = _resolve_in_sandbox(image_path)
#         return model.predict(str(resolved))
#     except Exception as e:
#         return f"Error: {str(e)}"


def inspect_segmentation_tool(mask_filename: str, ct_filename: str):
    """
    Analyzes a segmentation mask.
    ARGS:
        mask_filename: The name of the mask file (e.g., 'segmentation.nii.gz')
        ct_filename: The name of the original CT file (e.g., 'scan.nii.gz')
    """
    if analyzer is None:
        raise RuntimeError("Analyzer not initialized.")

    try:
        # Analyzer handles its own sandbox checks internally, but we can double check
        # or just rely on analyzer methods.
        metrics = analyzer.analyze_mask(mask_filename, ct_filename)
        image_name = analyzer.save_center_slice(mask_filename, ct_filename)
        
        return {
            "status": "success",
            "organ_metrics": metrics,
            "visual_file": image_name # The backend looks for this key!
        }
    except Exception as e:
        return f"Analysis Failed: {str(e)}"


def view_saved_image(image_filename: str):
    """
    Display any PNG saved in the sandbox (plots, visualizations, slice overlays).
    """
    try:
        _resolve_in_sandbox(image_filename)
    except Exception as e:
        return f"Error: {str(e)}"

    return {
        "status": "success",
        "visual_file": image_filename
    }

def list_sandbox_files():
    """
    Lists all files in the sandbox with detailed metadata. This should be used to explain what files are present or to get a better understanding of what files are available.
    For NIfTI files, it returns shape and voxel spacing to help identify the scan type.
    """
    if SANDBOX_ROOT is None:
        return "Error: Sandbox not initialized."
    
    try:
        files = [f for f in SANDBOX_ROOT.iterdir() if f.is_file() and not f.name.startswith('.')]
        
        inventory = []
        for f in files:
            # Basic File Stats
            size_mb = round(f.stat().st_size / (1024 * 1024), 2)
            item_info = {
                "filename": f.name,
                "size": f"{size_mb} MB",
                "type": "Unknown"
            }

            # NIfTI Specific Metadata (The "Smart" Part)
            if f.name.endswith((".nii.gz", ".nii")):
                try:
                    # Load header only (lazy loading, very fast)
                    img = cast(Any, nib_load(f))
                    shape = img.shape
                    zooms = img.header.get_zooms() # Voxel spacing (e.g., 1.5mm x 1.5mm x 3mm)
                    
                    item_info["type"] = "NIfTI Medical Image"
                    item_info["shape"] = str(shape)
                    item_info["spacing"] = f"({zooms[0]:.1f}, {zooms[1]:.1f}, {zooms[2]:.1f})"
                    
                    # Heuristic: Guess Anatomy based on dimensions
                    # Chest/Abdomen scans usually have ~512x512 in X/Y
                    if len(shape) == 3:
                        item_info["slices"] = shape[2]
                        if shape[2] > 100:
                            item_info["notes"] = "Large volume (likely full scan)"
                        else:
                            item_info["notes"] = "Small volume (likely crop or local)"
                            
                except Exception:
                    item_info["status"] = "Corrupted/Unreadable NIfTI"

            elif f.name.endswith((".png", ".jpg", ".jpeg")):
                item_info["type"] = "Visualization Image"

            inventory.append(item_info)
        
        if not inventory:
            return "The sandbox is empty."
            
        return inventory

    except Exception as e:
        return f"Error scanning sandbox: {str(e)}"
    

def search_web_medical(query: str):
    """
    Searches the web for medical knowledge, reference values, or guidelines.
    Use this to find normal organ sizes, Hounsfield Unit (HU) definitions, or clinical guidelines.
    """
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        
        # Format results nicely for the Agent to read
        formatted = ""
        for i, res in enumerate(results):
            formatted += f"Source {i+1}: {res['title']}\nSummary: {res['body']}\nURL: {res['href']}\n\n"
            
        return formatted
    except Exception as e:
        return f"Search failed: {str(e)}"
    

def run_python_analysis(code: str):
    """
    Executes a Python script to perform data analysis or visualization.
    The code will be executed in the sandbox directory.
    
    CRITICAL REQUIREMENTS FOR THE CODE:
    1. If you generate a plot, save it as a PNG file (e.g., plt.savefig('my_plot.png')).
    2. Do NOT use plt.show() (it will block the server).
    3. Use 'print()' to output text results.
    4. You have access to pandas, numpy, matplotlib, and nibabel.
    """
    if SANDBOX_ROOT is None:
        return "Error: Sandbox not initialized."
    
    # Create a specific filename for this script
    script_name = "agent_generated_analysis.py"
    script_path = SANDBOX_ROOT / script_name
    
    # Write the code to the file
    try:
        with open(script_path, "w") as f:
            f.write(code)
    except Exception as e:
        return f"Error writing script: {e}"

    # Execute the script in a subprocess inside the sandbox directory
    try:
        result = subprocess.run(
            [sys.executable, script_name], # Run with the same python interpreter
            capture_output=True,
            text=True,
            cwd=SANDBOX_ROOT, # CRITICAL: Run "inside" the sandbox
            timeout=30 # Safety timeout (seconds)
        )
        
        output = f"Execution Successful.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        return output

    except subprocess.TimeoutExpired:
        return "Error: Execution timed out (limit 30s)."
    except Exception as e:
        return f"Execution failed: {str(e)}"
