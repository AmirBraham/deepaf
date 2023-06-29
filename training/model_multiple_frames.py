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
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from tempfile import TemporaryDirectory
import warnings
import wandb

# Ignorer les avertissements
warnings.filterwarnings("ignore")

TRAIN_SIZE = 2000
TEST_SIZE = 1000
VAL_SIZE = 10000
BATCH_SIZE = 32
NUMBER_FRAMES = 5
NUM_EPOCHS = 10

######### DATA ############

wandb.login(key="59f93da2cc54b1f88fbb5aceb4e7f0e1fd7b983f")

# start a new wandb run to track this script
wandb.init(
    dir="./wandb",
    # set the wandb project where this run will be logged
    project="training_5_frames_backward",
    
    # track hyperparameters and run metadata
    config={
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE
    }
)

data_dir = '.'
data_file = "dataset.csv"
dataset_sizes = {"train":TRAIN_SIZE,"test":TEST_SIZE,"val":VAL_SIZE}

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
        if (l_fake == "0"):
            video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx,3], self.video_labels.iloc[idx, 0])
        else:
            video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx, 3], self.video_labels.iloc[idx, 0])

        video_frames,_,_ = read_video(video_path)
        num_frames = len(video_frames)

        #frame_index = random.randint(0,num_frames-1-self.number_frames)
        frames = [video_frames[-1 -i] for i in range(self.number_frames)]

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

        return frames_transformed, label

image_dataset_train = CustomImageDataset(annotations_file = data_file, video_dir = data_dir, status = "train", total_number=TRAIN_SIZE,number_frames=NUMBER_FRAMES)
image_dataset_test = CustomImageDataset(annotations_file = data_file, video_dir = data_dir, status = "test", total_number = TEST_SIZE,number_frames=NUMBER_FRAMES)
image_dataset_val = CustomImageDataset(annotations_file = data_file, video_dir = data_dir, status = "val", total_number = VAL_SIZE,number_frames=NUMBER_FRAMES)
image_dataloader_train = DataLoader(image_dataset_train,batch_size=BATCH_SIZE,shuffle=True, num_workers=4)
image_dataloader_test = DataLoader(image_dataset_test,batch_size=BATCH_SIZE,shuffle=True, num_workers=4)
image_dataloader_val = DataLoader(image_dataset_val,batch_size=1,shuffle=True, num_workers=4)

dataloaders = {"train" : image_dataloader_train, "test" : image_dataloader_test, "val" : image_dataloader_val}

###### MODEL #######
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

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
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
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
                            temp_loss += criterion(outputs, labels)

                        loss = temp_loss/num_frames
                        preds = torch.round(temp_preds/num_frames)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * batch_size
                    running_corrects += torch.sum(preds == labels.data)
                    
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                #print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                wandb.log({"acc":epoch_acc,"loss":epoch_loss})
                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

######## TRAINING ########

wandb.watch(model_conv,log_freq=100)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=NUM_EPOCHS)

print(model_conv)
# Spécifiez le chemin de sauvegarde souhaité
model_path = 'model_5_frames_backward.pt'

# Sauvegarde du modèle
torch.save(model_conv.state_dict(), model_path)

wandb.finish()

####### VISUALIZATION ########

