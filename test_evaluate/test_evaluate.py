from PIL import Image
from torchvision import transforms
import torchvision
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

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

class_names = {0 : "Fake", 1 : "Real"}

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imsave("test.png", inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    # Afficher la taille de l'image
    width, height = img.size
    print("Largeur :", width)
    print("Hauteur :", height)

    # Vérifier le mode de l'image (nombre de canaux)
    mode = img.mode
    print("Mode de l'image :", mode)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = data_transforms['test'](img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0].item()]}')
        print(class_names[preds[0].item()])
        imshow(img.cpu().data[0])

        model.train(mode=was_training)

# Spécifiez le chemin du modèle sauvegardé
model_path = '../model.pt'

# Créez une instance du modèle avec la même architecture que celle utilisée lors de l'entraînement
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Chargez les poids du modèle à partir du fichier
model.load_state_dict(torch.load(model_path))

# Mettez le modèle en mode d'évaluation
model.eval()

visualize_model_predictions(model,"deepfake-example_fake.png")

