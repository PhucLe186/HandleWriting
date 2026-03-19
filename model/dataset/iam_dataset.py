import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from until.resize_image import resize_with_aspect_show


class IAMDataset(Dataset):
    def __init__(self, hf_dataset, char2idx, img_height=32, img_width=512):
        self.data = hf_dataset
        self.char2idx = char2idx
        self.img_height = img_height
        self.img_width = img_width

        self.transform = transforms.Compose([
            # transforms.Grayscale(),
            # transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image'].convert('L')
        image = resize_with_aspect_show(image, 32, 512)
        image = self.transform(image)  # (1, H, W)

        label = sample['text']
        encoded = [self.char2idx[c] for c in label if c in self.char2idx]

        return image, torch.tensor(encoded, dtype=torch.long), label

def collate_fn(batch):
    images, labels, texts = zip(*batch)

    images = torch.stack(images, 0)  # (B, 1, H, W)

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)

    return images, labels_concat, label_lengths, texts

