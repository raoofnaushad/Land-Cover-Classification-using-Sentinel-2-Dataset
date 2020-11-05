import torch
from torchvision.transforms import transforms

import config
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def get_image(path):
    img = Image.open(path)
    # img.show()
    transform = transforms.Compose([
                                    transforms.Resize(size=(224, 224)),
                                    # transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    
    img = transform(img)
    return img




def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
            
    def __len__(self):
        return len(self.dl)
    