import torch 
from torch import nn
from torchvision.models import mobilenet_v3_small 

NUM_CLASSES = 26

model = mobilenet_v3_small() 
model.classifier[3] = nn.Linear(in_features=1024, out_features=NUM_CLASSES) 