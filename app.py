from statistics import mode
from flask import Flask, render_template, request

import torch
import torchvision
import cv2
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import numpy as np
from PIL import Image

app = Flask(__name__)

model=torchvision.models.regnet_y_32gf()

weights=torch.load('insect_model.pth',map_location=torch.device('cpu'))
model.fc=torch.nn.Linear(3712,142)
model.load_state_dict(weights,strict=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
model.eval()


def get_prediction(PATH_TO_IMAGE):
    image=cv2.imread(PATH_TO_IMAGE)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    result=evaluate(model,image)
    return result   

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Maybe a page about how it works?"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = get_prediction(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)

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
    common_file=open('common_names.txt', 'r')
    classes=[]
    common_classes = []
    content=file.readlines()
    common_content = common_file.readlines()
    for i in content:
        spl=i.split('\n')[0]
        classes.append(spl)
    for i in common_content:
        spl=i.split('\n')[0]
        common_classes.append(spl)
    with torch.inference_mode():
            image = image.to(device, non_blocking=True)
            output = model(image)
            op = torch.nn.functional.softmax(output)
            op= torch.argmax(op)
            return common_classes[op]


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)