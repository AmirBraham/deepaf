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
import wandb

torch.multiprocessing.set_sharing_strategy('file_system')

# Ignorer les avertissements
warnings.filterwarnings("ignore")

MIN_FRAMES_PER_VIDEO = 50
NUMBER_FRAMES = 5

data_dir = '.'

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
        'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, video_dir, status="train", total_number=math.inf, number_frames = 1,transform=data_transforms, target_transform=None):
        df = pd.read_csv(annotations_file)
        self.video_labels = df[df["status"] == status]
        video_labels_0 = self.video_labels[self.video_labels["label"] == 0]
        video_labels_1 = self.video_labels[self.video_labels["label"] == 1]
        
        num_samples = min(total_number // 2, len(video_labels_0), len(video_labels_1))

        video_labels_0 = video_labels_0.sample(num_samples)
        video_labels_1 = video_labels_1.sample(num_samples)

        self.video_labels = pd.concat([video_labels_0, video_labels_1], ignore_index=True)[:total_number]
        self.video_dir = video_dir
        self.transform = transform[status]
        self.target_transform = target_transform
        self.number_frames = number_frames
        print(self.video_labels)

    def __len__(self):
        return len(self.video_labels)

    def __getitem__(self, idx):
        l_status = self.video_labels.iloc[idx,2]
        l_fake = self.video_labels.iloc[idx,1]
        subfolder = self.video_labels.iloc[idx,3]
        file_name = self.video_labels.iloc[idx, 0]
        if (l_fake == "0"):
            video_path = os.path.join(self.video_dir, subfolder, file_name)
        else:
            video_path = os.path.join(self.video_dir, subfolder, file_name)

        video_frames,_,_ = read_video(video_path)
        num_frames = len(video_frames)

        frame_index = random.randint(0,num_frames-1-self.number_frames)
        frames = [video_frames[frame_index + i] for i in range(self.number_frames)]

        frames_transformed = []
        for frame in frames:
            # Convertir la frame en objet PIL
            frame_pil = Image.fromarray(frame.numpy().astype('uint8')).convert('RGB')
            frame_pil.save("working_frame.png")

            # Appliquer les transformations d'images à la frame
            if self.transform:
                frames_transformed.append(self.transform(frame_pil))
            else:
                frames_transformed.append(frame_pil)

        label = self.video_labels.iloc[idx, 1]

        if self.target_transform:
            label = self.target_transform(label)

        return frames_transformed, label, (subfolder + "/" + file_name)

image_dataset_test = CustomImageDataset(annotations_file = "dataset.csv", video_dir = data_dir, status = "val",total_number=1400,number_frames=NUMBER_FRAMES)
dataloaders = DataLoader(image_dataset_test,batch_size=1,shuffle=True, num_workers=2)


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
            inputs = inputs
            labels = labels.to(device)

            num_frames = len(inputs)
            batch_size = inputs[0].size(0)

            temp_loss = 0
            temp_preds = torch.Tensor([0 for i in range(batch_size)]).to(device)

            for index in range(num_frames):
                inputs_ieme_frame = inputs[index]
                inputs_ieme_frame = inputs_ieme_frame.to(device)
                outputs = model(inputs_ieme_frame)
                _, preds = torch.max(outputs, 1)
                temp_preds = torch.add(temp_preds,preds)

            preds = torch.round(temp_preds/num_frames)
            print(preds)

            for prediction, label in zip(preds, labels):
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

model_path = 'model_5_frames_backward.pt'
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
    print("FALSE POSITIVE:", file_path)
    destination_folder = "test_evaluate/false_positives"
    shutil.copy(file_path[0], destination_folder)

for input, prediction, file_path in false_negatives:
    print("FALSE NEGATIVE",file_path)
    destination_folder = "test_evaluate/false_negatives"
    shutil.copy(file_path[0], destination_folder)




