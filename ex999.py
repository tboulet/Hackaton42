import torch
from torch import optim, nn
from torchvision import models, transforms
from tqdm import tqdm
import numpy as np
import os
import cv2
from os.path import join
from NN.feature_extractor import *
from sklearn.cluster import KMeans
import pandas as pd


# code from https://towardsdatascience.com/image-feature-extraction-using-pytorch-e3b327c3607a

def obtain_features(dirct):
	files = os.listdir(dirct)
	files.sort()
	features = []
	i = 0
	for item in files:
		if i == 50:
			break
		print(item)
		if os.path.isfile(join(dirct, item)):
			img = cv2.imread(dirct + '/' + item)
			img = transform(img)
			img = img.reshape(1, 3, 448, 448)
			img = img.to(device)
			with torch.no_grad():
				feature = new_model(img)
		features.append(feature.cpu().detach().numpy().reshape(-1))
		i += 1
	return features

# Get pre-trained model
model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(448),
  transforms.ToTensor()                              
])

# Path to humans folders
path = "datasets/999_humans/humans/"
X_labeled_dir = join(path, "labeled")
X_unlabeled_dir = join(path, "unlabeled")
X_val_dir = join(path, "val")

features_x_lab = obtain_features(X_labeled_dir)
features_x_val = obtain_features(X_val_dir)

features_x_lab = np.array(features_x_lab)
features_x_val = np.array(features_x_val)

# Clustering
model = KMeans(n_clusters=2, random_state=42)
model.fit(features_x_lab)
labels = model.labels_

print(labels)


sample_submission = pd.read_csv('sample_submission.csv')
new_submission = sample_submission
new_submission['label'] = labels
new_submission.to_csv('submission_1.csv', index=False, header=False)