import torch, os, random, numpy as np, albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from thop import profile
from model import DeepLabV3PlusMobile
from dataset import CombinedVOC

class HybridLoss(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        probs, valid_mask = torch.nn.functional.softmax(logits, dim=1), (target!= 255).float()
        target_one_hot = torch.nn.functional.one_hot(torch.clamp(target, 0, 20), 21).permute(0, 3, 1, 2).float()
        inter = (probs * target_one_hot * valid_mask.unsqueeze(1)).sum(dim=(2, 3))
        union = (probs + target_one_hot).sum(dim=(2, 3)) * valid_mask.unsqueeze(1).sum(dim=(2, 3))
        return ce_loss + (1 - ((2. * inter + 1e-6) / (union + 1e-6)).mean())

def run_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_dir = os.getcwd()
    path = os.path.join(current_dir, 'VOC2012_train_val/VOC2012_train_val')


   
    img_names = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(path, "JPEGImages")) if f.endswith('.jpg')]
    random.shuffle(img_names)
    split = int(0.8 * len(img_names))
    
    aug = A.Compose([
        A.Resize(320, 320),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    train_loader = DataLoader(CombinedVOC(path, img_names[:split], aug), batch_size=16, shuffle=True)
    val_loader = DataLoader(CombinedVOC(path, img_names[split:], aug), batch_size=8)

    model = DeepLabV3PlusMobile().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion, scaler = HybridLoss(torch.ones(21).to(device)), GradScaler()

    best_dice = 0
    for epoch in range(170):
        model.train(); epoch_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(imgs)
                loss = criterion(output, masks)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); epoch_loss += loss.item()

        model.eval(); dice_all =[]
        with torch.no_grad():
            for imgs, masks in val_loader:
                preds = torch.argmax(model(imgs.to(device)), dim=1)
                for c in range(21):
                    p, t = (preds == c).float(), (masks.to(device) == c).float()
                    dice_all.append(((2.*(p*t).sum()+1e-6)/(p.sum()+t.sum()+1e-6)).item())
        
        avg_dice = np.mean(dice_all)
        flops, _ = profile(model, inputs=(torch.randn(1, 3, 320, 320).to(device),), verbose=False)
        print(f"Epoch {epoch+1:02} | Loss: {epoch_loss/len(train_loader):.4f} | Dice: {avg_dice:.4f} | GFLOPs: {flops/1e9:.3f}", flush=True)
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), 'best_model.pt')
        scheduler.step()

if __name__ == "__main__": run_training()