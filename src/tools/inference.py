import os
import torch
import traceback
from monai.bundle import ConfigParser
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import SaveImage, EnsureChannelFirstd, Compose

MODEL_REGISTRY = {}


class MedicalModel:
    def __init__(self, name: str, bundle_name: str, root_dir: str, device: str = "cuda"):
        self.name = name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.bundle_path = os.path.join(root_dir, bundle_name)
        self.config_path = os.path.join(self.bundle_path, "configs", "inference.json")

        self.parser = ConfigParser()
        self.parser.read_config(self.config_path)

        self.pre_transforms = self.parser.get_parsed_content("preprocessing")
        self.post_transforms = self.parser.get_parsed_content("postprocessing")

        # Laod model
        print(f"⚡️ Loading model '{name}' to {self.device} ")
        self.network = self.parser.get_parsed_content("network").to(self.device)

        # Load weights
        ckpt_path = os.path.join(self.bundle_path, "models" , "model.pt")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.network.load_state_dict(checkpoint.get("model", checkpoint))
            print(f"    ✅ Weights loaded for '{name}' model.")
        else:
            print(f"    No checkpoint found at {ckpt_path} for '{name}' model.")
        
        self.network.eval()
        self.inferer = self.parser.get_parsed_content("inferer")

    def predict(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
    
        output_dir = os.path.dirname(file_path)

        data = [{"image": file_path}]
        ds = Dataset(data=data, transform=self.pre_transforms)
        loader = DataLoader(ds, batch_size=1)

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                outputs = self.inferer(inputs=images, network=self.network)
                batch["pred"] = outputs
                
                items = decollate_batch(batch)
                for item in items:
                    # Post-processing
                    for t in self.post_transforms.transforms:
                        if "Save" in type(t).__name__:
                            continue
                        item = t(item)
                    
                    mask_tensor = item["pred"]

                    # if multi chennl, flatten
                    if mask_tensor.shape[0] > 1:
                        mask_tensor = torch.argmax(mask_tensor, dim=0, keepdim=True)
                    
                    # Make sure mask is uint8
                    if mask_tensor.dtype != torch.uint8:
                        mask_tensor = (mask_tensor > 0.5).to(torch.uint8)
                    
                    # Save resuults
                    saver = SaveImage(
                        output_dir=output_dir,
                        output_postfix="seg",
                        output_ext=".nii.gz",
                        separate_folder=False,
                        print_log=False,
                        output_dtype=torch.uint8
                    )

                    meta = item.get("pred_meta_dict") or item.get("image_meta_dict")
                    saver(mask_tensor, meta_data=meta)
        
        return f"Segmentation complete. Mask saved in {output_dir}"
            

def load_models_from_hydra(cfg):
    root = cfg.medical_models.bundle_root
    device = cfg.medical_models.device
    for friendly_name, bundle_name in cfg.medical_models.active_bundles.items():
        try:
            model = MedicalModel(friendly_name, bundle_name, root, device)
            MODEL_REGISTRY[friendly_name] = model
        except Exception as e:
            print(f"Nope: Failed to load model '{friendly_name}': {e}")
            traceback.print_exc()

def get_model(name: str):
    return MODEL_REGISTRY.get(name)


# class MedicalModel:
#     def __init__(self, name: str, bundle_name: str, root_dir: str, device: str = "cuda"):
#         self.name = name
#         self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
#         self.bundle_path = os.path.join(root_dir, bundle_name)
#         self.config_path = os.path.join(self.bundle_path, "configs", "inference.json")

#         self.parser = ConfigParser()
#         self.parser.read_config(self.config_path)

#         # --- Patch Config for Brain Model (Channel Last) ---
#         if "brain" in name.lower():
#             print(f"🔧 Patching {name} config for Channel-Last data...")
#             raw_preprocessing = self.parser.config.get("preprocessing")
            
#             transforms_list = None
#             if isinstance(raw_preprocessing, dict) and "transforms" in raw_preprocessing:
#                 transforms_list = raw_preprocessing["transforms"]
#             elif isinstance(raw_preprocessing, list):
#                 transforms_list = raw_preprocessing
            
#             if transforms_list is not None:
#                 found = False
#                 load_image_index = 0
                
#                 # 1. Scan the list
#                 for i, step in enumerate(transforms_list):
#                     target = step.get("_target_", "")
                    
#                     # Track where the Loader is (we must insert AFTER this)
#                     if "LoadImage" in target:
#                         load_image_index = i
                    
#                     # Check if EnsureChannelFirstd already exists
#                     if "EnsureChannelFirstd" in target:
#                         print("   -> Found EnsureChannelFirstd, updating channel_dim to -1")
#                         step["channel_dim"] = -1
#                         found = True
                
#                 # 2. If not found, insert it specifically AFTER the loader
#                 if not found:
#                     insert_idx = load_image_index + 1
#                     print(f"   -> EnsureChannelFirstd missing. Inserting at index {insert_idx} (after Loader).")
#                     transforms_list.insert(insert_idx, {
#                         "_target_": "monai.transforms.EnsureChannelFirstd",
#                         "keys": ["image"],
#                         "channel_dim": -1
#                     })

#         self.pre_transforms = self.parser.get_parsed_content("preprocessing")
#         self.post_transforms = self.parser.get_parsed_content("postprocessing")

#         self.network = self.parser.get_parsed_content("network").to(self.device)
        
#         # --- Robust Checkpoint Loading ---
#         ckpt_path = os.path.join(self.bundle_path, "models", "model.pt")
#         if os.path.exists(ckpt_path):
#             try:
#                 checkpoint = torch.load(ckpt_path, map_location=self.device)
#                 if isinstance(checkpoint, dict):
#                     for key in ("model", "state_dict"):
#                         if key in checkpoint:
#                             checkpoint = checkpoint[key]
#                             break
#                 self.network.load_state_dict(checkpoint, strict=False)
#                 print(f"⚡ Loaded weights for '{name}'")
#             except Exception as e:
#                 print(f"❌ Error loading weights for '{name}': {e}")
#         else:
#             print(f"⚠️ No checkpoint found at {ckpt_path}")
            
#         self.network.eval()
#         self.inferer = self.parser.get_parsed_content("inferer")

#     def predict(self, file_path: str) -> str:
#         if not os.path.exists(file_path):
#             return f"Error: File not found at {file_path}"

#         output_dir = os.path.dirname(file_path)
        
#         data = [{"image": file_path}]
#         dataset = Dataset(data=data, transform=self.pre_transforms)
#         loader = DataLoader(dataset, batch_size=1, num_workers=0)
        
#         try:
#             with torch.no_grad():
#                 for batch in loader:
#                     images = batch["image"].to(self.device)

#                     outputs = self.inferer(inputs=images, network=self.network)
#                     batch["pred"] = outputs

#                     items = decollate_batch(batch)
#                     for item in items:
#                         # 1. Run MONAI post-processing (Invert, standard thresholds)
#                         for t in self.post_transforms.transforms:
#                             if "Save" in type(t).__name__:
#                                 continue
#                             item = t(item)

#                         # 2. --- RECONSTRUCTION LOGIC ---
#                         pred = item["pred"]  # Shape is usually (3, X, Y, Z)
                        
#                         # Create a blank Integer canvas (background = 0)
#                         # We use uint8 (Byte) to save space and correct format
#                         mask_map = torch.zeros(pred.shape[1:], dtype=torch.uint8, device=self.device)

#                         if "brain" in self.name.lower() and pred.shape[0] == 3:
#                             # BraTS Specific Hierarchy:
#                             # Channel 0: Tumor Core (TC)
#                             # Channel 1: Whole Tumor (WT)
#                             # Channel 2: Enhancing Tumor (ET)
                            
#                             # Threshold to binary (0 or 1) if not already
#                             if pred.is_floating_point():
#                                 pred = (pred > 0.5)

#                             # Paint Step 1: Whole Tumor -> Label 2 (Edema)
#                             # We paint the largest region first
#                             mask_map[pred[1] == 1] = 2
                            
#                             # Paint Step 2: Tumor Core -> Label 1 (Necrotic/Non-Enhancing)
#                             # This overwrites the Edema inside the core
#                             mask_map[pred[0] == 1] = 1
                            
#                             # Paint Step 3: Enhancing Tumor -> Label 3 (Enhancing)
#                             # This overwrites the Necrotic region inside the enhancing part
#                             mask_map[pred[2] == 1] = 3
                            
#                             # Add the channel dimension back: (1, X, Y, Z)
#                             final_mask = mask_map.unsqueeze(0)
                            
#                         elif pred.shape[0] > 1:
#                             # Fallback for other multi-class models (argmax)
#                             final_mask = torch.argmax(pred, dim=0, keepdim=True).to(torch.uint8)
#                         else:
#                             # Binary models (Spleen)
#                             final_mask = (pred > 0.5).to(torch.uint8)

#                         # 3. Save as clean UINT8
#                         saver = SaveImage(
#                             output_dir=output_dir,
#                             output_postfix="seg",
#                             output_ext=".nii.gz",
#                             separate_folder=False,
#                             print_log=False,
#                             output_dtype=torch.uint8 # Explicitly force NIfTI data type
#                         )

#                         # Use metadata to preserve physical orientation
#                         meta = item.get("pred_meta_dict") or item.get("image_meta_dict")
#                         saver(final_mask, meta_data=meta)
                    
#             return f"Segmentation complete. Mask saved in {output_dir}"

#         except Exception as e:
#             print("\n❌ INFERENCE ERROR:")
#             traceback.print_exc()
#             return f"Technical Error: {str(e)}"

# def load_models_from_hydra(cfg):
#     root = cfg.medical_models.bundle_root
#     device = cfg.medical_models.device
#     for friendly_name, bundle_name in cfg.medical_models.active_bundles.items():
#         try:
#             model = MedicalModel(friendly_name, bundle_name, root, device)
#             MODEL_REGISTRY[friendly_name] = model
#         except Exception as e:
#             print(f"❌ Failed to load model '{friendly_name}': {e}")
#             traceback.print_exc()

# def get_model(name: str):
#     return MODEL_REGISTRY.get(name)