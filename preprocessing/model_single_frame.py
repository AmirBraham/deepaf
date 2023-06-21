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
import random
from torch.utils.data import DataLoader
from PIL import Image

min_frames_per_video = 150

data_dir = './dataset-real'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, video_dir, status="train", total_number=math.inf, transform=data_transforms, target_transform=None):
        df = pd.read_csv(annotations_file)
        self.video_labels = df[df["status"] == status]
        self.video_labels = self.video_labels.sample(frac=1).reset_index(drop=True)[:min(len(self.video_labels), total_number)]
        self.video_dir = video_dir
        self.transform = transform[status]
        self.target_transform = target_transform
        self.frame_indices = self._get_random_frame_indices(len(self.video_labels))
        print(self.video_labels)

    def __len__(self):
        return len(self.video_labels)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx, 2], self.video_labels.iloc[idx, 0])

        video_frames,_,_ = read_video(video_path)
        frame_index = self.frame_indices[idx]
        frame = video_frames[frame_index]

        # Convertir la frame en objet PIL
        frame_pil = Image.fromarray(frame.permute(1, 2, 0).numpy().astype('uint8'))

        # Appliquer les transformations d'images à la frame
        if self.transform:
            frame_transformed = self.transform(frame_pil)
        else:
            frame_transformed = frame_pil

        label = self.video_labels.iloc[idx, 1]

        if self.target_transform:
            label = self.target_transform(label)

        return frame_transformed, label

    def _get_random_frame_indices(self, num_videos):
        frame_indices = []
        for _ in range(num_videos):
            frame_indices.append(random.randint(0, min_frames_per_video))  # Choix aléatoire d'une frame
        return frame_indices

image_dataset_train = CustomImageDataset(annotations_file = "./dataset-real/dataset.csv", video_dir = data_dir, status = "train", total_number=300)
image_dataset_test = CustomImageDataset(annotations_file = "./dataset-real/dataset.csv", video_dir = data_dir, status = "test", total_number = 300)
image_dataloader_train = DataLoader(image_dataset_train,batch_size=4,shuffle=True, num_workers=4)
image_dataloader_test = DataLoader(image_dataset_test,batch_size=4,shuffle=True, num_workers=4)

inputs, classes = next(iter(image_dataloader_train))
print(inputs[0])
