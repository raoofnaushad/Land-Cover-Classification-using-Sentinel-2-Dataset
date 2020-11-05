from dataset import to_device, get_device, get_image
from model import get_model
import torch
import config



def predict_single():
    image = get_image(config.PATH)
    device = get_device()
    model = get_model()
    model.eval()
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    preds = model(xb)
    _, prediction = torch.max(preds.cpu().detach(), dim=1)
    return decode_target(int(prediction), text_labels=True)



def decode_target(target, text_labels=True):
    result = []
    if text_labels:
        return config.idx_class_labels[target]
    else:
        return target
    