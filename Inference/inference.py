import argparse, os, cv2, torch, numpy as np
from PIL import Image
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import DeepLabV3PlusMobile
from thop import profile

def calculate_dice(pred, target):
    """Calculates Binary Dice while ignoring VOC void class (255)."""
    mask = (target!= 255)
    p = (pred > 0).astype(np.float32)
    t = (target > 0).astype(np.float32)
    intersection = np.sum(p * t * mask)
    union = np.sum(p * mask) + np.sum(t * mask)
    return (2. * intersection + 1e-6) / (union + 1e-6)

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True, help="Path to test images")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to save masks")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load Model (Architecture synced to node configuration '3' and '13')
    model = DeepLabV3PlusMobile(num_classes=21).to(device)
    state_dict = torch.load('best_model.pt', map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 2. Complexity Analysis
    dummy_input = torch.randn(1, 3, 320, 320).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)

    # 3. Preprocessing (ImageNet Norm is required for MobileNetV3)
    
    aug = A.Compose([
        A.Resize(320, 320),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 4. Correct Directory Detection (Handles standard VOC or flat folders)
    image_source = args.in_dir
    if os.path.exists(os.path.join(args.in_dir, "JPEGImages")):
        image_source = os.path.join(args.in_dir, "JPEGImages")
    
    gt_dir = os.path.join(args.in_dir, "SegmentationClass")
    has_gt = os.path.exists(gt_dir)

    img_files = [f for f in os.listdir(image_source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n[INFO] Found {len(img_files)} images in {image_source}")

    dice_scores =[]
    for filename in img_files:
        img_path = os.path.join(image_source, filename)
        orig_img = Image.open(img_path).convert('RGB')
        w, h = orig_img.size
        
        # Inference
        img_tensor = aug(image=np.array(orig_img))['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            # Upscale back to original size before argmax for 'perfect' boundaries
            output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # 5. BINARY MASK GENERATION (Crucial Step)
        # Combine all classes 1-20 into foreground (255), background is 0
        binary_mask = (pred > 0).astype(np.uint8) * 255
        
        # 6. FILE SAVING (Fixed Naming Bug)
        # Requirement: Filename must be IDENTICAL to input filename
        out_path = os.path.join(args.out_dir, filename)
        
        # Use cv2.imwrite for reliable binary saving
        success = cv2.imwrite(out_path, binary_mask)
        
        if success:
            print(f"Generating binary mask: {filename} -> {out_path}")
        else:
            print(f" Failed to write file to {out_path}. check permissions.")

        # 7. Validation Logic
        if has_gt:
            gt_path = os.path.join(gt_dir, os.path.splitext(filename) + ".png")
            if os.path.exists(gt_path):
                dice_scores.append(calculate_dice(pred, np.array(Image.open(gt_path))))

    # 8. Final Report
    print("\n" + "="*50)
    print(f"Model Complexity: {flops / 1e9:.3f} GFLOPs")
    if dice_scores:
        print(f"Mean Dice Score: {np.mean(dice_scores):.4f}")
    print("="*50)

if __name__ == "__main__":
    run_inference()