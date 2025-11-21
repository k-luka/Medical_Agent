import os
from monai.bundle import download

def main():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    print("🚀 Downloading MONAI Bundles... (This saves the 'Brains' of your agent)")

    # 1. Spleen Model (CT)
    # This creates: models/spleen_ct_segmentation/configs/inference.json
    print("\n--- Downloading Spleen Model ---")
    download(
        name="spleen_ct_segmentation", 
        bundle_dir=model_dir,
        version="0.1.0" # Pinning version for stability
    )

    # 2. Brain Tumor Model (MRI)
    # This creates: models/brats_mri_segmentation/configs/inference.json
    print("\n--- Downloading Brain Tumor Model ---")
    download(
        name="brats_mri_segmentation", 
        bundle_dir=model_dir,
        version="0.1.0"
    )

    print("\n✅ Done! You should now see a 'models' folder in your project.")



if __name__ == "__main__":
    main()