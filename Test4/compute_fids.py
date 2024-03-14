import torch
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as v2
import os

class normalize2(nn.Module):
    def forward(self, img):
        img = ((img - img.min())/ (img.max() - img.min()))
        q = torch.stack([img, img, img], dim=0)
        return q.squeeze(1)

class StrongLensingDataset(Dataset):
    def __init__(self, imgs, train=True, transform=None):
        self.imgs = imgs
        self.len = sum(1 for _, _, files in os.walk(self.imgs) for f in files)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx += 1
        path = os.path.join(self.imgs, f"sample{idx}.npy")
        img = torch.from_numpy(np.load(path))
        img = img.to(torch.float32)

        if self.transform:
            img = self.transform(img)

        return img


def get_dataloader(file_path, image_size=160, train=True, batch_size=16, num_workers=2):
    transform = v2.Compose(
        [
            v2.Resize((image_size, image_size), antialias=True), # closest multiple of 32 to 155
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.0615], std=[0.1164]), # mean = 0, variance = 1
        ]
    )
    transform2 = v2.Compose(
        [
            v2.Resize((image_size, image_size), antialias=True), # closest multiple of 32 to 155
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.0615], std=[0.1164]), # mean = 0, variance = 1
            normalize2(),
        ]
    )
    if train:
        dataset = StrongLensingDataset(file_path, train=train, transform=transform)
        dataloader = DataLoader(
            dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
    else:
        dataset = StrongLensingDataset(file_path, train=train, transform=transform2)
        dataloader = DataLoader(
            dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
    return dataloader

def get_images(path, ema=False, num=1200):
    path = os.path.join(path, "ema" if ema else "normal")
    images = []
    for i in range(1, num+1):
        img = np.load(f"{path}/out{i}.npy")
        img = (img - img.min())/ (img.max() - img.min())
        img = torch.tensor(img)
        img = torch.stack([img, img, img], dim=0)
        images.append(img.unsqueeze(0))
    images = torch.cat(images)
    return images

real_images = next(iter(get_dataloader("dataset-ddpm/val", train=False, batch_size=1000, num_workers=0)))
fake_images_normal = get_images("output-images", ema=False)
fake_images_ema = get_images("output-images", ema=True)

#real_images = (real_images * 255).to(torch.uint8)
#fake_images_normal = (fake_images_normal * 255).to(torch.uint8)
#fake_images_ema = (fake_images_ema * 255).to(torch.uint8)

#fid_normal = FrechetInceptionDistance(feature=2048, normalize=False).set_dtype(torch.float32)
#fid_ema = FrechetInceptionDistance(feature=2048, normalize=False).set_dtype(torch.float32)

#fid_normal.update(real_images, real=True)
#fid_normal.update(fake_images_normal, real=False)

#fid_ema.update(real_images, real=True)
#fid_ema.update(fake_images_ema, real=False)


#print(f"Normal: {fid_normal.compute()}")
#print(f"EMA: {fid_ema.compute()}")

w = 160
h = 160

fig = plt.figure(figsize=(24, 24))
columns = 16
rows = 16
for i in range(1, columns * rows + 1):
    img = np.load(f"output-images/ema/out{i}.npy")
    ax = fig.add_subplot(rows, columns, i)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(img)
plt.savefig('ema.png', bbox_inches='tight')
plt.show()
