import torch
import torchvision
import cv2
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from PIL import Image



def transforms_validation(image):
    crop_size=224
    resize_size=256
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    interpolation=InterpolationMode.BILINEAR
    transforms_val = transforms.Compose(
                    [
                    transforms.Resize(resize_size, interpolation=interpolation),
                    transforms.CenterCrop(crop_size),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=mean, std=std)])
    image = Image.fromarray(np.uint8(image))
    image=transforms_val(image).reshape((1,3,224,224))
    return image

def evaluate(model,image):
    model.eval()
    device=torch.device('cpu')
    image=transforms_validation(image)
    file=open('classes.txt','r')
    classes=[]
    content=file.readlines()
    for i in content:
        spl=i.split('\n')[0]
        classes.append(spl)
    
    with torch.inference_mode():
            image = image.to(device, non_blocking=True)
            output = model(image)
            op = torch.nn.functional.softmax(output)
            op= torch.argmax(op)
            return classes[op] 
