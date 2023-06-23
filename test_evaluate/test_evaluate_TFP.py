from PIL import Image
from torchvision import transforms
import torchvision
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_video
import math
import pandas as pd
import warnings
import shutil


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

# Ignorer les avertissements
warnings.filterwarnings("ignore")

MIN_FRAMES_PER_VIDEO = 50

data_dir = '.'

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, video_dir, status="train", total_number=math.inf, transform=data_transforms, target_transform=None):
        df = pd.read_csv(annotations_file)
        self.video_labels = df[df["status"] == status]
        video_labels_0 = self.video_labels[self.video_labels["label"] == 0]
        video_labels_1 = self.video_labels[self.video_labels["label"] == 1]
        
        num_samples = min(total_number // 2, len(video_labels_0), len(video_labels_1))

        video_labels_0 = video_labels_0.sample(num_samples)
        video_labels_1 = video_labels_1.sample(num_samples)

        self.video_labels = pd.concat([video_labels_0, video_labels_1], ignore_index=True)[:total_number]
        #self.video_labels = self.video_labels.sample(frac=1).reset_index(drop=True)[:min(len(self.video_labels), total_number)]
        self.video_dir = video_dir
        self.transform = transform[status]
        self.target_transform = target_transform
        self.frame_indices = self._get_random_frame_indices(len(self.video_labels))
        print(self.video_labels)

    def __len__(self):
        return len(self.video_labels)

    def __getitem__(self, idx):
        l_status = self.video_labels.iloc[idx,2]
        l_fake = self.video_labels.iloc[idx,1]
        if (l_fake == "0"):
            video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx,3], self.video_labels.iloc[idx, 0])
        else:
            video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx, 3], self.video_labels.iloc[idx, 0])

        video_frames,_,_ = read_video(video_path)
        frame_index = self.frame_indices[idx]
        frame = video_frames[frame_index]
        frame.size()

        # Convertir la frame en objet PIL
        frame_pil = Image.fromarray(frame.numpy().astype('uint8')).convert('RGB')
        frame_pil.save("working_frame.png")

        # Appliquer les transformations d'images à la frame
        if self.transform:
            frame_transformed = self.transform(frame_pil)
        else:
            frame_transformed = frame_pil

        label = self.video_labels.iloc[idx, 1]
        subfolder = self.video_labels.iloc[idx,3]
        file_name = self.video_labels.iloc[idx,0]

        if self.target_transform:
            label = self.target_transform(label)

        return frame_transformed, label, (subfolder + "/" + file_name)

    def _get_random_frame_indices(self, num_videos):
        frame_indices = []
        for _ in range(num_videos):
            frame_indices.append(random.randint(0, MIN_FRAMES_PER_VIDEO))  # Choix aléatoire d'une frame
        return frame_indices

image_dataset_test = CustomImageDataset(annotations_file = "dataset.csv", video_dir = data_dir, status = "val",total_number=2000)
dataloaders = DataLoader(image_dataset_test,batch_size=1,shuffle=True, num_workers=4)


def visualize_model_predictions(model, dataloader):
    was_training = model.training
    model.eval()

    true_positives = []
    false_positives = []
    true_negatives = []
    false_negatives = []
    total = 0

    with torch.no_grad():
        for inputs, labels, file_path in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            for prediction, label in zip(predictions, labels):
                if prediction == 1 and label == 1:
                    true_positives.append((inputs, prediction,file_path))
                elif prediction == 1 and label == 0:
                    false_positives.append((inputs, prediction,file_path))
                elif prediction == 0 and label == 0:
                    true_negatives.append((inputs, prediction,file_path))
                elif prediction == 0 and label == 1:
                    false_negatives.append((inputs, prediction,file_path))
                total += 1

    model.train(mode=was_training)

    return true_positives, false_positives, true_negatives, false_negatives, total

model_path = 'model.pt'
model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

true_positives, false_positives, true_negatives, false_negatives, total = visualize_model_predictions(model, dataloaders)
print("Nombre de VRAIS POSITIFS : ", len(true_positives), " soit : ", 100*len(true_positives)/total, "%")
print("Nombre de FAUX POSITIFS : ", len(false_positives), " soit : ", 100*len(false_positives)/total, "%")
print("Nombre de VRAIS NEGATIFS : ", len(true_negatives), " soit : ", 100*len(true_negatives)/total, "%")
print("Nombre de FAUX NEGATIFS : ", len(false_negatives)," soit : ", 100*len(false_negatives)/total, "%")
print("Nombre Total : ", total)

def imsave(inp, file_path, title=None):
    """Display image for Tensor."""
    inp = inp.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    std_expanded = np.expand_dims(std, axis=(1, 2))
    inp = std_expanded * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imsave(file_path, inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

for input, prediction, file_path in false_positives:
    print(file_path)
    destination_folder = "test_evaluate/false_positives"
    shutil.copy(file_path[0], destination_folder)

for input, prediction, file_path in false_negatives:
    print(file_path)
    destination_folder = "test_evaluate/false_negatives"
    shutil.copy(file_path[0], destination_folder)




