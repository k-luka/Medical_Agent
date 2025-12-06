import os
import glob
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

# CONFIG
DATA_ROOT = "data/HaN-Seg/set_1"
OUTPUT_ROOT = "data/HaN-Seg/processed_ready_for_train"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Same OAR list as your code
OAR_NAMES = [
    "A_Carotid_L", "A_Carotid_R", "Arytenoid", "Bone_Mandible", "Brainstem",
    "BuccalMucosa", "Cavity_Oral", "Cochlea_L", "Cochlea_R", "Cricopharyngeus",
    "Esophagus_S", "Eye_AL", "Eye_AR", "Eye_PL", "Eye_PR", "Glnd_Lacrimal_L",
    "Glnd_Lacrimal_R", "Glnd_Submand_L", "Glnd_Submand_R", "Glnd_Thyroid",
    "Glottis", "Larynx_SG", "Lips", "OpticChiasm", "OpticNrv_L", "OpticNrv_R",
    "Parotid_L", "Parotid_R", "Pituitary", "SpinalCord"
]

def read_clean_image(path):
    """
    Reads an image and forces it to be 3D Float32.
    Handles 4D images with a singleton dimension (common in NRRD/NIfTI).
    """
    img = sitk.ReadImage(path)
    
    # Fix dimensions: if 4D, convert to 3D
    if img.GetDimension() == 4:
        # Check if the 4th dimension is actually 1 (singleton)
        if img.GetSize()[3] == 1:
            # Slice out the first volume: [x, y, z, 0]
            img = img[:, :, :, 0]
        else:
            # If it's a true 4D sequence, we usually just want the first timepoint for anatomy
            print(f"Warning: {path} is 4D {img.GetSize()}. Using first volume.")
            img = img[:, :, :, 0]
            
    # Fix type: registration works best with Float32
    img = sitk.Cast(img, sitk.sitkFloat32)
    
    return img

def register_mri_to_ct(fixed_image, moving_image):
    # Ensure both images are 3D before creating the transform
    # (This double-check prevents the specific error you saw)
    if fixed_image.GetDimension() != 3 or moving_image.GetDimension() != 3:
        raise ValueError(f"Dimension mismatch! CT: {fixed_image.GetDimension()}D, MRI: {moving_image.GetDimension()}D")

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Gradient Descent is robust, but LBFGSB is often faster/better for rigid
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, 
        numberOfIterations=100, 
        estimateLearningRate=registration_method.EachIteration
    )
    
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    # Resample MRI to match CT space exactly
    resampled_mri = sitk.Resample(
        moving_image, fixed_image, final_transform, 
        sitk.sitkLinear, 0.0, moving_image.GetPixelID()
    )
    return resampled_mri

def process_case(case_id):
    case_dir = os.path.join(DATA_ROOT, case_id)
    
    # PATHS
    ct_path = os.path.join(case_dir, f"{case_id}_IMG_CT.nrrd")
    mr_path = os.path.join(case_dir, f"{case_id}_IMG_MR_T1.nrrd")
    
    # Load CT & MRI using the new safe reader
    # ---------------------------------------------
    ct_img = read_clean_image(ct_path)
    
    if os.path.exists(mr_path):
        mr_img = read_clean_image(mr_path)
        # Now registration will work because dimensions match
        mr_reg = register_mri_to_ct(ct_img, mr_img)
    else:
        print(f"Warning: No MRI for {case_id}")
        return

    # Merge labels (logic remains mostly the same, just ensure 3D output)
    label_map_np = np.zeros(sitk.GetArrayFromImage(ct_img).shape, dtype=np.uint8)
    
    for idx, oar in enumerate(OAR_NAMES, start=1):
        seg_path = os.path.join(case_dir, f"{case_id}_OAR_{oar}.seg.nrrd")
        if os.path.exists(seg_path):
            # Load mask and also force 3D
            seg_img = read_clean_image(seg_path)
            
            # Resample seg to CT space
            seg_resampled = sitk.Resample(
                seg_img, ct_img, sitk.Transform(), 
                sitk.sitkNearestNeighbor, 0.0, seg_img.GetPixelID()
            )
            seg_arr = sitk.GetArrayFromImage(seg_resampled)
            label_map_np[seg_arr > 0] = idx

    final_label = sitk.GetImageFromArray(label_map_np)
    final_label.CopyInformation(ct_img)

    # Save outputs
    sitk.WriteImage(ct_img, os.path.join(OUTPUT_ROOT, f"{case_id}_ct.nii.gz"))
    sitk.WriteImage(mr_reg, os.path.join(OUTPUT_ROOT, f"{case_id}_mr.nii.gz"))
    sitk.WriteImage(final_label, os.path.join(OUTPUT_ROOT, f"{case_id}_label.nii.gz"))
    
# RUN LOOP
case_folders = sorted([d for d in os.listdir(DATA_ROOT) if d.startswith("case_")])
for case in tqdm(case_folders):
    process_case(case)
