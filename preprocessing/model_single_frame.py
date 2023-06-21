import io
import os
import math
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.io import read_video
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader

class CustomVideoDataset(Dataset):
    def __init__(self, annotations_file, video_dir, status = "train", total_number = math.inf, transform=None, target_transform=None):
        df = pd.read_csv(annotations_file)
        self.video_labels = df[df["status"] == status]
        self.video_labels = self.video_labels.sample(frac=1).reset_index(drop=True)[:min(len(self.video_labels),total_number)]
        self.video_dir = video_dir
        self.transform = transform
        self.target_transform = target_transform
        print(self.video_labels)

    def __len__(self):
        return len(self.video_labels)

    def __getitem__(self, idx):

        video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx, 2], self.video_labels.iloc[idx, 0])
        video = read_video(video_path)[0]
        label = self.video_labels.iloc[idx, 1]
        if self.transform:
            video = self.transform(video)
        if self.target_transform:
            label = self.target_transform(label)
        return video, label

video_datasets_train = CustomVideoDataset(annotations_file = "dataset.csv", video_dir = "./vox",total_number=300)
video_datasets_test = CustomVideoDataset(annotations_file = "dataset.csv", video_dir = "./vox", status = "test",total_number=300)
video_dataloader_train = DataLoader(video_datasets_train)
video_dataloader_test = DataLoader(video_datasets_test,batch_size = 2)
print(len(video_dataloader_test))
print(len(video_dataloader_train))

inputs, classes = next(iter(video_dataloader_train))
plt.imshow(inputs[0][17])