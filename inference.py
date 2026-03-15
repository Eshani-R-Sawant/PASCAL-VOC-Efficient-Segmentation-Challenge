import argparse, os, cv2, torch, numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import DeepLabV3PlusMobile

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    model = DeepLabV3PlusMobile(num_classes=21).to(device)
    # Setting strict=False ignores the "total_ops" and "total_params" keys
    state_dict = torch.load('best_model.pt', map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

   
    aug = A.Compose([
        A.Resize(320, 320),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    for filename in os.listdir(args.in_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(args.in_dir, filename)
            orig = Image.open(img_path).convert('RGB')
            w, h = orig.size
            
            img_tensor = transform(image=np.array(orig))['image'].unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                output = torch.nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # Binary conversion: Background (0) -> 0, Foreground (1-20) -> 255
            binary_mask = (pred > 0).astype(np.uint8) * 255
            out_filename = os.path.splitext(filename) + ".png"
            Image.fromarray(binary_mask).save(os.path.join(args.out_dir, out_filename))

if __name__ == "__main__":
    run_inference()