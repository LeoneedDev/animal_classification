import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


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
