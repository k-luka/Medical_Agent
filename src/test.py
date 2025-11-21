import nibabel as nib
import os

# Check a Spleen File
spleen_path = "./data/Task09_Spleen/imagesTr/spleen_2.nii.gz"
if os.path.exists(spleen_path):
    img = nib.load(spleen_path)
    print(f"Spleen Shape: {img.shape} (Should be 3D)")

# Check a Brain File
brain_path = "./data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz"
if os.path.exists(brain_path):
    img = nib.load(brain_path)
    print(f"Brain Shape: {img.shape} (Should be 4D with 4 channels)")