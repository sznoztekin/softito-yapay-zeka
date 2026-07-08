import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
from PIL import Image

# 1. Saf CNN Mimarisi
class BasitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Özellik çıkarıcı katmanlar
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 128x128 -> 64x64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 64x64 -> 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)  # 32x32 -> 16x16
        )
        # Sınıflandırıcı katmanlar
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 2) # 2 sınıf: 0=İnsan Yok, 1=İnsan Var
        )
    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

# 2. Veri Seti Okuyucu
class BasitDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = sorted(os.listdir(os.path.join(root, "PNGImages")))
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        img = Image.open(img_path).convert("RGB").resize((128, 128))
        # Etiket: Bu veri setindeki tüm resimlerde insan olduğu için 1 (İnsan Var) diyoruz
        return T.ToTensor()(img), torch.tensor(1, dtype=torch.long)
    def __len__(self): return len(self.imgs)

# 3. Eğitim Döngüsü
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BasitCNN().to(device)
dataset = BasitDataset('.') # Bulunduğun klasörü kullanır
loader = DataLoader(dataset, batch_size=4, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Saf CNN eğitimi başlıyor...")
model.train()
for epoch in range(5):
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} tamamlandı. Loss: {loss.item():.4f}")

print("Eğitim tamamlandı!")