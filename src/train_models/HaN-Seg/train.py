import torch
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader, CacheDataset
from sklearn.model_selection import train_test_split
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import os

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CropForegroundd,
    RandSpatialCropd, ScaleIntensityRanged, ScaleIntensityRangePercentilesd,
    RandFlipd, RandAffined, EnsureTyped, ConcatItemsd, DeleteItemsd
)

def get_train_transforms():
    return Compose([
        # Load raw data
        LoadImaged(keys=["image_ct", "image_mr", "label"]),
        
        # SAFELY add channel dim (B, C, D, H, W)
        EnsureChannelFirstd(keys=["image_ct", "image_mr", "label"], channel_dim="no_channel"),

        # Normalize CT (absolute physics values)
        # CT is always consistent: -150 is soft tissue, 1000 is bone.
        ScaleIntensityRanged(
            keys=["image_ct"], 
            a_min=-150, a_max=250, 
            b_min=0.0, b_max=1.0, 
            clip=True
        ),

        # Normalize MRI (adaptive relative values)
        # This finds the bottom 0.5% and top 99.5% of intensities and scales them to 0-1.
        # It handles your 35,000 max value automatically.
        ScaleIntensityRangePercentilesd(
            keys=["image_mr"], 
            lower=0.5, upper=99.5, 
            b_min=0.0, b_max=1.0, 
            clip=True, 
            relative=False
        ),

        # Combine into multi-modal input (Channel 0=CT, Channel 1=MRI)
        ConcatItemsd(keys=["image_ct", "image_mr"], name="image", dim=0),
        DeleteItemsd(keys=["image_ct", "image_mr"]),

        # Spatial transforms (crop, zoom, rotate)
        # Note: We crop on 'image' (the combined pair) so CT/MRI stay aligned
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=(192, 192, 96),
            random_center=True, random_size=False
        ),
        
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandAffined(
            keys=["image", "label"], prob=0.3, 
            rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1), 
            mode=("bilinear", "nearest")
        ),

        EnsureTyped(keys=["image", "label"]),
    ])

def get_val_transforms():
    """
    Validation transforms:
    - Same loading and normalization as training.
    - No random crops (sliding window on the full volume).
    - No random flips/rotations.
    """
    return Compose([
        # Load
        LoadImaged(keys=["image_ct", "image_mr", "label"]),
        EnsureChannelFirstd(keys=["image_ct", "image_mr", "label"], channel_dim="no_channel"),

        # Normalize CT
        ScaleIntensityRanged(
            keys=["image_ct"], a_min=-150, a_max=250, 
            b_min=0.0, b_max=1.0, clip=True
        ),

        # Normalize MRI (same percentiles as training)
        ScaleIntensityRangePercentilesd(
            keys=["image_mr"], lower=0.5, upper=99.5, 
            b_min=0.0, b_max=1.0, clip=True, relative=False
        ),

        # Concat
        ConcatItemsd(keys=["image_ct", "image_mr"], name="image", dim=0),
        DeleteItemsd(keys=["image_ct", "image_mr"]),

        # Crop foreground (optional but recommended to remove empty air)
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        # Note: NO RandSpatialCropd here! We validate on the full (cropped) volume.
        EnsureTyped(keys=["image", "label"]),
    ])

def create_dataloaders(batch_size=4, num_workers=4, val_split=0.2):
    """
    - Get the list of files.
    - Split into train/val.
    - Create MONAI datasets and PyTorch dataloaders.
    """
    full_data_list = build_datalist()
    
    # Split
    train_files, val_files = train_test_split(
        full_data_list, test_size=val_split, random_state=42, shuffle=True
    )
    
    print(f"Total Cases: {len(full_data_list)}")
    print(f"Training: {len(train_files)} cases")
    print(f"Validation: {len(val_files)} cases")

    # Create Datasets
    train_ds = CacheDataset(data=train_files, transform=get_train_transforms(), num_workers=num_workers, cache_rate=1.0)
    val_ds = CacheDataset(data=val_files, transform=get_val_transforms(), num_workers=num_workers, cache_rate=1.0)

    # Create Loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, prefetch_factor=4,
    )
    
    # (sliding window handles the rest)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=True, prefetch_factor=4,
    )
    
    return train_loader, val_loader


# This must match the output folder from your preprocess.py
PROCESSED_ROOT = "data/HaN-Seg/processed_ready_for_train"

def build_datalist():
    """
    Scans the processed folder and creates a list of dictionaries
    for the dataloader.
    """
    # We search for files ending in '_ct.nii.gz' to identify cases
    case_ids = sorted([
        f.split('_ct.nii.gz')[0] 
        for f in os.listdir(PROCESSED_ROOT) if f.endswith('_ct.nii.gz')
    ])
    
    data_list = []
    for cid in case_ids:
        data_list.append({
            "image_ct": os.path.join(PROCESSED_ROOT, f"{cid}_ct.nii.gz"),
            "image_mr": os.path.join(PROCESSED_ROOT, f"{cid}_mr.nii.gz"), 
            "label":    os.path.join(PROCESSED_ROOT, f"{cid}_label.nii.gz"),
        })
    return data_list


# MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROI_SIZE = (192, 192, 96)  # Used for sliding window inference
MAX_EPOCHS = 500
VAL_INTERVAL = 5
LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    # UPDATED: Removed 'img_size' based on your correct observation
    model = SwinUNETR(
        in_channels=2, 
        out_channels=31, 
        feature_size=48, 
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True, 
    ).to(DEVICE)
    return model

def train_one_epoch(model, loader, optimizer, scheduler, loss_func):
    model.train()
    epoch_loss = 0
    step = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch in pbar:
        # Move data to GPU
        inputs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(inputs)
        
        # Calculate Loss
        loss = loss_func(outputs, labels)
        
        # Backward Pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        step += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    scheduler.step()
    return epoch_loss / step

def validate(model, loader, dice_metric):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            val_inputs = batch["image"].to(DEVICE)
            val_labels = batch["label"].to(DEVICE)
            
            # SLIDING WINDOW INFERENCE
            # Stitches patches together to evaluate the FULL volume
            val_outputs = sliding_window_inference(
                inputs=val_inputs, 
                roi_size=ROI_SIZE, 
                sw_batch_size=4, 
                predictor=model,
                overlap=0.25
            )
            
            # Convert Raw Logits -> One-Hot Discrete for Dice
            # Argmax gets the most likely class index (0-30)
            # We add a channel dim back to match MONAI metric expectation
            val_outputs = torch.argmax(val_outputs, dim=1, keepdim=True)
            
            # Compute Dice
            dice_metric(y_pred=val_outputs, y=val_labels)
            
    # Aggregate the final mean dice score
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return mean_dice

def run_training():
    print(f"--- Starting HaN-Seg Training on {DEVICE} ---")
    train_loader, val_loader = create_dataloaders(batch_size=4, num_workers=4)
    
    # Setup model, loss, optimizer
    model = get_model()
    
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)

    parametersWithDecay = []
    parametersWithoutDecay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name.lower() or "positionalEmbedding" in name:
            parametersWithoutDecay.append(param)
        else:
            parametersWithDecay.append(param)
    parameterGroups = [
        {"params": parametersWithDecay, "weight_decay": cfg.model.optimizer.weight_decay},
        {"params": parametersWithoutDecay, "weight_decay": 0.0}
    ]

    optimizer = torch.optim.AdamW(parameterGroups, lr=LR, weight_decay=WEIGHT_DECAY)

    # scheduler
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = MAX_EPOCHS)
    scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=(MAX_EPOCHS // 50))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[MAX_EPOCHS // 50])

    # Metric: ignore background (index 0) so we measure organ performance only
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # Main loop
    best_metric = -1
    best_metric_epoch = -1

    for epoch in range(MAX_EPOCHS):
        print(f"Epoch {epoch+1}/{MAX_EPOCHS}")
        
        # TRAIN
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_function)
        print(f"  Average Loss: {train_loss:.4f}")

        # VALIDATE (Every N epochs)
        if (epoch + 1) % VAL_INTERVAL == 0:
            metric = validate(model, val_loader, dice_metric)
            print(f"  >>> Val Dice Score: {metric:.4f}")
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "models/HaN-Seg/best_model_swinunetr.pth")
                print("  >>> New Best Model Saved! <<<")

    print(f"Training Finished. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")

if __name__ == "__main__":
    run_training()
