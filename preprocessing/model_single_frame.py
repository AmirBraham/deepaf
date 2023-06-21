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
        frame_pil = Image.fromarray(frame[0].numpy().astype('uint8')).convert('RGB')

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

image_dataset_train = CustomImageDataset(annotations_file = "dataset.csv", video_dir = data_dir, status = "train", total_number=500)
image_dataset_test = CustomImageDataset(annotations_file = "dataset.csv", video_dir = data_dir, status = "test", total_number = 100)
image_dataloader_train = DataLoader(image_dataset_train,batch_size=4,shuffle=True, num_workers=4)
image_dataloader_test = DataLoader(image_dataset_test,batch_size=4,shuffle=True, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dict = {"train" : image_dataloader_train, "test" : image_dataloader_test}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model
