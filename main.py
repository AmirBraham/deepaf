import torch

model = torch.load('vox-cpk.pth.tar', map_location='cpu')
print(model)
