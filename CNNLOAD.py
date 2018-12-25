

from __future__ import print_function, division
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
plt.ion()   # interactive mode
from scipy.misc import imread, imresize



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load('skin_cancer_model.pth.tar'))
class_names = ['benign', 'malignant']



def predict_single_image(class_names, model_ft, image):
    img = imread(image)
    img = imresize(img, (224, 224))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    img = transform(img)  # (3, 224, 224)
    img = img.unsqueeze(0)  # (1, 3, 224, 224)
    model_ft.eval()
    with torch.no_grad():
        outputs = model_ft(img)
        _, preds = torch.max(outputs, 1)
        print("predicted: {}".format(class_names[preds]))
    return class_names[preds]

predict_single_image(class_names, model_ft, 'test.jpg')