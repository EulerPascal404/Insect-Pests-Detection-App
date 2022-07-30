from statistics import mode
from flask import Flask, render_template, request

import torch
import torchvision
import evaluate
import cv2
from torchvision.transforms.functional import InterpolationMode

app = Flask(__name__)

model=torchvision.models.regnet_y_32gf()

weights=torch.load('insect_model.pth',map_location=torch.device('cpu'))['model']
model.fc=torch.nn.Linear(3712,142)
model.load_state_dict(weights,strict=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
model.eval()

def  get_prediction(PATH_TO_IMAGE):
	image = cv2.imread(PATH_TO_IMAGE)
	result = evaluate.evaluate(model,image)
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


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)