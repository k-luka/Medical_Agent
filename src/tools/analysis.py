import base64
from pathlib import Path
from typing import Any, Dict, Optional, cast

import matplotlib.pyplot as plt
from nibabel.loadsave import load as nib_load
import numpy as np


class MedicalAnalyzer:
    """
    Lightweight mask analysis and visualization.
    Keeps all I/O inside a sandbox directory.
    """

    ORGAN_LABELS: Dict[int, str] = {
        1: "Spleen",
        2: "Right Kidney",
        3: "Left Kidney",
        4: "Gallbladder",
        5: "Esophagus",
        6: "Liver",
        7: "Stomach",
        8: "Aorta",
        9: "IVC",
        10: "Portal and Splenic Veins",
        11: "Pancreas",
        12: "Right adrenal gland",
        13: "Left adrenal gland",
    }

    def __init__(self, sandbox_dir: Path):
        self.sandbox_dir = sandbox_dir.resolve()
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def _get_safe_path(self, filename: str) -> Path:
        full_path = (self.sandbox_dir / filename).resolve()
        try:
            full_path.relative_to(self.sandbox_dir)
        except ValueError:
            raise PermissionError(f"Attempted access outside sandbox: {full_path}")
        if not full_path.exists():
            raise FileNotFoundError(f"File not found in sandbox: {filename}")
        return full_path

    def analyze_mask(self, mask_filename: str, original_ct_filename: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        mask_path = self._get_safe_path(mask_filename)

        mask_img = cast(Any, nib_load(mask_path))
        mask_data = np.asarray(mask_img.get_fdata(), dtype=np.int16)

        img_data = None
        if original_ct_filename:
            ct_path = self._get_safe_path(original_ct_filename)
            ct_img = cast(Any, nib_load(ct_path))
            img_data = np.asarray(ct_img.get_fdata(), dtype=np.float32)
            if img_data.shape != mask_data.shape:
                raise ValueError(f"CT/mask shape mismatch: {img_data.shape} vs {mask_data.shape}")

        zooms = tuple(float(z) for z in mask_img.header.get_zooms()[:3])
        voxel_vol_cc = float((zooms[0] * zooms[1] * zooms[2]) / 1000.0)

        report: Dict[str, Dict[str, float]] = {}
        for label_idx in np.unique(mask_data):
            if label_idx == 0:
                continue

            organ_mask = mask_data == label_idx
            voxel_count = int(np.sum(organ_mask))
            volume_cc = float(round(voxel_count * voxel_vol_cc, 2))

            organ_name = self.ORGAN_LABELS.get(int(label_idx), f"Organ_{int(label_idx)}")
            stats: Dict[str, float] = {"volume_cc": volume_cc}

            if img_data is not None:
                organ_pixels = img_data[organ_mask]
                stats["mean_HU"] = round(float(np.mean(organ_pixels)), 1)
                stats["std_HU"] = round(float(np.std(organ_pixels)), 1)

            report[organ_name] = stats

        return report

    def save_center_slice(self, mask_filename: str, original_ct_filename: str) -> str:
        mask_path = self._get_safe_path(mask_filename)
        ct_path = self._get_safe_path(original_ct_filename)

        mask_img = cast(Any, nib_load(mask_path))
        ct_img = cast(Any, nib_load(ct_path))

        mask_data = np.asarray(mask_img.get_fdata(), dtype=np.int16)
        img_data = np.asarray(ct_img.get_fdata(), dtype=np.float32)

        if mask_data.shape != img_data.shape:
            raise ValueError(f"CT/mask shape mismatch: {img_data.shape} vs {mask_data.shape}")

        # Pick the slice with the maximum labeled area to avoid empty-looking slices
        slice_sums = np.sum(mask_data > 0, axis=(0, 1))
        if slice_sums.max() == 0:
            raise ValueError("Empty mask provided.")
        z_idx = int(np.argmax(slice_sums))

        img_slice = np.rot90(img_data[:, :, z_idx])
        mask_slice = np.rot90(mask_data[:, :, z_idx])

        plt.figure(figsize=(6, 6))
        plt.imshow(img_slice, cmap="gray")
        plt.imshow(mask_slice, cmap="jet", alpha=0.45 * (mask_slice > 0))
        plt.axis("off")
        plt.title(f"Slice {z_idx}")

        base_name = mask_path.name.split(".")[0]  # avoid double extensions like .nii.gz
        output_filename = f"vis_{base_name}.png"
        output_path = self.sandbox_dir / output_filename
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        return output_filename

    def read_slice_png(self, image_filename: str) -> str:
        """
        Return a base64-encoded PNG (for agents that only consume text).
        """
        image_path = self._get_safe_path(image_filename)
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return encoded
