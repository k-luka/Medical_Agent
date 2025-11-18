

def segment_image(image_path: str, target: str,):
    return {
        "image_path": image_path,
        "target": target,
        "mask_path": f"{image_path}.fake_mask.nii.gz",
        "volume_ml": 12.3,
    }

