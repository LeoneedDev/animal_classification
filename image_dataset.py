import torch
from torch.utils.data import Dataset
from torch import Tensor
from PIL import Image
import pandas as pd
# import cv2
# import numpy as np
# from PIL import Image
# from torchvision import transforms


# class ImageDataset(Dataset):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                              std=[0.5, 0.5, 0.5])
#     ])

#     def __init__(self, paths):
#         self.paths = paths

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         path = self.paths[idx]
#         img_np = self.load_and_clean_image(path)
#         x = self.to_tensor(img_np)
#         return x

#     def load_and_clean_image(self, path):
#         img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
#         if img is None:
#             raise ValueError("Failed to read image")

#         if img.dtype == np.uint16:
#             img = (img.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
#         elif img.dtype == np.float32 or img.dtype == np.float64:
#             img = np.clip(img, 0.0, 1.0)
#             img = (img * 255.0).astype(np.uint8)

#         if len(img.shape) == 2:
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         else:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         img = cv2.medianBlur(img, 3)

#         return img

#     def to_tensor(self, img_np):
#         img_pil = Image.fromarray(img_np)
#         tensor = self.transform(img_pil)
#         return tensor

#     def preprocess_for_model(self, path):
#         img_np = self.load_and_clean_image(path)
#         tensor = self.to_tensor(img_np)
#         return tensor


class ImageDataset(Dataset):
    def __init__(self, x: pd.Series, y: pd.Series, transform=None):
        self.data = pd.DataFrame()
        self.data['image_path'] = x.copy()
        self.data['label_encoded'] = y.copy()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        y_label = torch.tensor(int(self.data.iloc[idx]['label_encoded']))

        return (image, y_label)
