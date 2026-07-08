import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import os

# Modeli oluştur (Eğitimdeki yapıyla birebir aynı olmalı)
def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model = get_model(num_classes=2)
model.load_state_dict(torch.load("mask_rcnn_egitilmis.pth", map_location=device))
model.eval()

# Resmi test et
test_resmi = "test_resmi.jpg"
if os.path.exists(test_resmi):
    img = Image.open(test_resmi).convert("RGB")
    img_tensor = T.ToTensor()(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(img_tensor)
        
    print(f"Tahmin tamamlandı! Tespit edilen nesne sayısı: {len(prediction[0]['boxes'])}")
else:
    print(f"{test_resmi} bulunamadı!")