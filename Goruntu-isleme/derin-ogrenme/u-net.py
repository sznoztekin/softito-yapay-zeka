import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import os
from PIL import Image
import numpy as np

# --- U-Net Mimarisi ---
class ConvBlok(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.blok = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.blok(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = ConvBlok(3, 64); self.pool = nn.MaxPool2d(2)
        self.down2 = ConvBlok(64, 128)
        self.bottleneck = ConvBlok(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.final = nn.Conv2d(64, 1, kernel_size=1) # 1 sınıf (insan)

    def forward(self, x):
        c1 = self.down1(x); p1 = self.pool(c1)
        c2 = self.down2(p1); p2 = self.pool(c2)
        b = self.bottleneck(p2)
        u1 = self.up1(b); u1 = torch.cat([u1, c2], dim=1); u1 = ConvBlok(256, 128)(u1) # Basitleştirilmiş skip
        u2 = self.up2(u1); u2 = torch.cat([u2, c1], dim=1); u2 = ConvBlok(128, 64)(u2)
        return self.final(u2)

# --- Veri Seti Okuyucu ---
class UNetDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = sorted(os.listdir(os.path.join(root, "PNGImages")))
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, "PNGImages", self.imgs[idx])).convert("RGB").resize((128,128))
        mask = Image.open(os.path.join(self.root, "PedMasks", self.imgs[idx].replace(".png", "_mask.png"))).resize((128,128))
        return TF.to_tensor(img), torch.tensor(np.array(mask) > 0, dtype=torch.float32).unsqueeze(0)
    def __len__(self): return len(self.imgs)

# --- Eğitim ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
dataset = UNetDataset('.')
loader = DataLoader(dataset, batch_size=2, shuffle=True)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("U-Net Eğitimi Başlıyor...")
model.train()
for epoch in range(5):
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} tamamlandı. Loss: {loss.item():.4f}")