import json
import logging
import traceback
from pathlib import Path
import numpy as np
import torch
from typing import Any, Dict, List, cast

from monai.bundle.config_parser import ConfigParser
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.utils import decollate_batch 
from monai.transforms.compose import Compose
from monai.transforms.io.array import SaveImage


# Configure logging
logger = logging.getLogger(__name__)

# --- 1. Base Class (Shared Logic) ---
class MedicalModel:
    def __init__(self, name: str, bundle_name: str, root_dir: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.bundle_path = Path(root_dir) / bundle_name
        self.config_path = self.bundle_path / "configs" / "inference.json"

        if not self.config_path.exists():
            raise FileNotFoundError(f"Bundle config not found at {self.config_path}")

        self.parser = ConfigParser()
        self.parser.read_config(str(self.config_path))

        self.network = self._load_network()
        self.inferer = self.parser.get_parsed_content("inferer")
        self.pre_transforms = self.parser.get_parsed_content("preprocessing")
        self.post_transforms = self._load_post_transforms()

    def _load_network(self) -> torch.nn.Module:
        print(f"⚡️ Loading model '{self.name}'...")
        # Patch for 'img_size' error in older bundles
        network_config = self.parser.get("network")
        if "img_size" in network_config:
            network_config.pop("img_size")
            self.parser["network"] = network_config

        network = self.parser.get_parsed_content("network").to(self.device)
        
        ckpt_path = self.bundle_path / "models" / "model.pt"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            network.load_state_dict(state_dict)
        else:
            print(f"⚠️ Warning: No checkpoint found for {self.name}")
        
        network.eval()
        return network

    def _load_post_transforms(self) -> Compose:
        transforms = self.parser.get_parsed_content("postprocessing")
        return transforms if isinstance(transforms, Compose) else Compose([])

    # Explicit return type 'str' fixes the inheritance conflict
    def predict(self, file_path: str) -> str:
        raise NotImplementedError("Subclasses must implement predict()")


# --- 2. Segmentation Subclass ---
class SegmentationModel(MedicalModel):
    def _load_post_transforms(self) -> Compose:
        transforms = self.parser.get_parsed_content("postprocessing")
        if isinstance(transforms, Compose):
            # Remove SaveImage so we can handle saving manually
            clean = [t for t in transforms.transforms if "Save" not in type(t).__name__]
            return Compose(clean)
        return transforms

    def predict(self, file_path: str) -> str:
        path = Path(file_path)
        output_dir = path.parent
        
        # Prepare Data
        ds = Dataset(data=[{"image": str(path)}], transform=self.pre_transforms)
        loader = DataLoader(ds, batch_size=1, num_workers=0)

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                batch["pred"] = self.inferer(inputs=images, network=self.network)
                
                # --- FIX: Explicit Cast for Type Checker ---
                items = cast(List[Dict[str, Any]], decollate_batch(batch))

                for item in items:
                    self._save_mask(item, output_dir)
        
        return f"Segmentation complete. Mask saved in {output_dir}"

    def _save_mask(self, item: Dict[str, Any], output_dir: Path):
        if self.post_transforms:
            item = cast(Dict[str, Any], self.post_transforms(item))
        
        mask = item["pred"]
        # Argmax if multi-channel (Multi-organ)
        if mask.shape[0] > 1: 
            mask = torch.argmax(mask, dim=0, keepdim=True)
        
        # Ensure uint8
        if mask.dtype != torch.uint8:
             mask = mask.to(torch.uint8)

        saver = SaveImage(output_dir=str(output_dir), output_postfix="seg", output_ext=".nii.gz", 
                          separate_folder=False, print_log=False, output_dtype=np.uint8)
        saver(mask, meta_data=item.get("image_meta_dict"))


# --- 3. Detection Subclass ---
class DetectionModel(MedicalModel):
    def predict(self, file_path: str) -> str:
        path = Path(file_path)
        output_json = path.parent / f"{path.stem}_detection.json"
        
        ds = Dataset(data=[{"image": str(path)}], transform=self.pre_transforms)
        loader = DataLoader(ds, batch_size=1, num_workers=0)

        results = []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                
                # 1. Inference
                batch["pred"] = self.inferer(inputs=images, network=self.network)
                
                # --- FIX: Explicit Cast for Type Checker ---
                items = cast(List[Dict[str, Any]], decollate_batch(batch))

                # 3. Post-Processing
                for item in items:
                    if self.post_transforms:
                        item = cast(Dict[str, Any], self.post_transforms(item))
                    
                    # 4. Extract Results
                    if "box" in item and "label" in item and "score" in item:
                        boxes = item["box"].cpu().numpy().tolist()
                        scores = item["score"].cpu().numpy().tolist()
                        labels = item["label"].cpu().numpy().tolist()
                        
                        detections = []
                        for b, s, l in zip(boxes, scores, labels):
                            if s > 0.10: # Confidence Threshold
                                detections.append({
                                    "box": b, 
                                    "score": s, 
                                    "label": int(l)
                                })
                        
                        results.append({
                            "file": str(path),
                            "detections": detections
                        })

        # Save Report
        with open(output_json, "w") as f:
            json.dump(results, f, indent=4)
            
        return f"Detection complete. Found {len(results[0]['detections']) if results else 0} items. Saved to {output_json}"


# --- 4. Factory Logic ---
MODEL_REGISTRY = {}

def load_models_from_hydra(cfg):
    root = cfg.medical_models.bundle_root
    device = cfg.medical_models.device
    
    for friendly_name, bundle_name in cfg.medical_models.active_bundles.items():
        try:
            # Simple heuristic
            if "detection" in bundle_name.lower():
                model = DetectionModel(friendly_name, bundle_name, root, device)
            else:
                model = SegmentationModel(friendly_name, bundle_name, root, device)
                
            MODEL_REGISTRY[friendly_name] = model
            print(f"✅ Registered '{friendly_name}' as {type(model).__name__}")
        except Exception as e:
            print(f"❌ Failed to load '{friendly_name}': {e}")
            traceback.print_exc()

def get_model(name: str):
    return MODEL_REGISTRY.get(name)

