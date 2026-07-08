import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np

# 1. Kendi Veri Seti Okuyucumuz (Dış kütüphaneye bağımlı değildir)
class PennFudanDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:] 
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        image = torchvision.transforms.functional.to_tensor(img)
        target = {"boxes": boxes, "labels": labels, "masks": masks}
        return image, target

    def __len__(self):
        return len(self.imgs)

# 2. Model Kurulumu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)
model.to(device)

# 3. Veri Yükleyici ve Eğitim
# 'PennFudanPed' klasörünü arama, doğrudan bulunduğun klasöre bak
dataset = PennFudanDataset('.')
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
model.train()

print("Eğitim başlıyor...")
for epoch in range(1):
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 5 == 0:
            print(f"Epoch {epoch} | Batch {i} | Loss: {losses.item():.4f}")

print("Eğitim tamamlandı!")

# ... eğitim döngüsünün bittiği yer ...
print(f"Epoch {epoch} | Batch {i} | Loss: {losses.item():.4f}")

# Eğitimi bitirdikten sonra modeli kaydet:
torch.save(model.state_dict(), "mask_rcnn_egitilmis.pth")
print("Model başarıyla kaydedildi!")