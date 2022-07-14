from torch.utils.data import Dataset
import torch
from PIL import Image

class CAR_Dataset(torch.utils.data.Dataset):
    def __init__(self, filelist, target, transforms=None):
        self.transforms = transforms
        self.imgs = filelist
        self.target = target
        
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        target = torch.as_tensor(self.target[idx], dtype=torch.float32)
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
    

class CAR_Testset(torch.utils.data.Dataset):
    def __init__(self, filelist, transforms=None):
        self.transforms = transforms
        self.imgs = filelist
        
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.imgs)