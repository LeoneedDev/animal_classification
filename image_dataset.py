from torch.utils.data import Dataset
from PIL import Image
from torch import Tensor
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

from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
