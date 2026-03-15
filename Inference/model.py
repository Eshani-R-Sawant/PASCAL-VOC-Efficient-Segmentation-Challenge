import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from thop import profile
import numpy as np
import os, random, cv2
from PIL import Image

# --- 1. LIGHTWEIGHT ATTENTION: ECA-NET ---
class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((np.log2(channels) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)

# --- 2. ARCHITECTURE ---
class AtrousSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.gn = nn.GroupNorm(8, out_ch) 
        self.act = nn.Hardswish(inplace=True)
    def forward(self, x): return self.act(self.gn(self.pointwise(self.depthwise(x))))

class DeepLabV3PlusMobile(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        self.backbone = create_feature_extractor(base, return_nodes={'3': 'low', '12': 'high'})
        
        self.project_high = nn.Sequential(
            nn.Conv2d(576, 256, 1, bias=False), 
            nn.GroupNorm(8, 256), 
            nn.ReLU(),
            ECAModule(256)
        )
        
        self.aspp_branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 128, 1, bias=False), nn.GroupNorm(8, 128), nn.ReLU()),
            AtrousSeparableConv(256, 128, dilation=6, padding=6),
            AtrousSeparableConv(256, 128, dilation=12, padding=12),
            AtrousSeparableConv(256, 128, dilation=18, padding=18),
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(256, 128, 1, bias=False), nn.ReLU())
        ])
        
        self.aspp_fuse = nn.Sequential(nn.Conv2d(128*5, 256, 1, bias=False), nn.GroupNorm(8, 256), nn.ReLU())
        self.low_level_project = nn.Sequential(nn.Conv2d(24, 48, 1, bias=False), nn.GroupNorm(8, 48), nn.ReLU())
        self.decoder = nn.Sequential(AtrousSeparableConv(256 + 48, 128), nn.Conv2d(128, num_classes, 1))

    def forward(self, x):
        feat = self.backbone(x)
        low, high = feat['low'], feat['high']
        high = self.project_high(high)
        aspp_res = []
        for b in self.aspp_branches:
            res = b(high)
            if res.shape[2:] != high.shape[2:]:
                res = F.interpolate(res, size=high.shape[2:], mode='bilinear', align_corners=False)
            aspp_res.append(res)
        high = F.interpolate(self.aspp_fuse(torch.cat(aspp_res, dim=1)), size=low.shape[2:], mode='bilinear', align_corners=False)
        return F.interpolate(self.decoder(torch.cat([high, self.low_level_project(low)], dim=1)), size=x.shape[2:], mode='bilinear', align_corners=False)

# --- 3. DATASET ---
class CombinedVOC(torch.utils.data.Dataset):
    def __init__(self, base_path, img_list, transform=None):
        self.img_dir = os.path.join(base_path, "JPEGImages")
        self.gt_dir = os.path.join(base_path, "SegmentationClass")
        self.sam_dir = os.path.join(base_path, "SAM_Segmentation_Maps1")
        self.img_list, self.transform = img_list, transform

    def __getitem__(self, idx):
        name = self.img_list[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, f"{name}.jpg")).convert('RGB'))
        gt_path = os.path.join(self.gt_dir, f"{name}.png")
        mask = np.array(Image.open(gt_path)) if os.path.exists(gt_path) else np.array(Image.open(os.path.join(self.sam_dir, f"{name}.png")))
        mask[mask == 255] = 0
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        if not torch.is_tensor(img):
            img = torch.from_numpy(img).permute(2, 0, 1)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)
        return img, mask.long()
    def __len__(self): return len(self.img_list)

# --- 4. LOSS ---
class HybridLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        target_one_hot = F.one_hot(torch.clamp(target, 0, 20), 21).permute(0, 3, 1, 2).float()
        intersection = (probs * target_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice_loss = 1 - ((2. * intersection + 1e-6) / (union + 1e-6)).mean()
        return self.ce(logits, target) + dice_loss

# --- 5. TRAINING ---
def run_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = "/scratch/m25cse012/DL_Assignment_PSSI_COPY/DL_Mini_project/VOC2012_train_val/VOC2012_train_val"
    img_names = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(path, "JPEGImages")) if f.endswith('.jpg')]
    random.shuffle(img_names)
    split = int(0.8 * len(img_names))
    
    # Universal Resize
    aug = A.Compose([A.Resize(320, 320), ToTensorV2()])
    train_loader = DataLoader(CombinedVOC(path, img_names[:split], aug), batch_size=16, shuffle=True)
    val_loader = DataLoader(CombinedVOC(path, img_names[split:], aug), batch_size=8)

    weights = torch.tensor([1.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.0, 1.8, 1.5, 2.5, 2.0, 2.2, 2.0, 2.0, 2.0, 1.0, 2.5, 2.0, 2.5, 2.0, 2.0]).to(device)
    model = DeepLabV3PlusMobile().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = HybridLoss(weights)

    for epoch in range(100):
        model.train(); epoch_loss = 0
        for imgs, masks in train_loader:
            # FIX: Ensure float type and normalization
            imgs = imgs.to(device).float() / 255.0 
            masks = masks.to(device)
            
            scale = random.uniform(0.7, 1.3)
            imgs_s = F.interpolate(imgs, scale_factor=scale, mode='bilinear', align_corners=False)
            masks_s = F.interpolate(masks.unsqueeze(1).float(), scale_factor=scale, mode='nearest').squeeze(1).long()
            
            optimizer.zero_grad()
            loss = criterion(model(imgs_s), masks_s)
            loss.backward(); optimizer.step(); epoch_loss += loss.item()

        model.eval(); dice_all = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device).float() / 255.0
                preds = torch.argmax(model(imgs), dim=1)
                for c in range(21):
                    p, t = (preds == c).float(), (masks.to(device) == c).float()
                    dice_all.append(((2.*(p*t).sum()+1e-6)/(p.sum()+t.sum()+1e-6)).item())
        
        flops, _ = profile(model, inputs=(torch.randn(1, 3, 320, 320).to(device),), verbose=False)
        print(f"Epoch {epoch+1:02} | Loss: {epoch_loss/len(train_loader):.4f} | Dice: {np.mean(dice_all):.4f} | GFLOPs: {flops/1e9:.3f}")

if __name__ == "__main__":
    run_training()