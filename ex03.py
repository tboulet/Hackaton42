from NN.model_for_mnist import *
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

def calculateDistance(img, imgref1, imgref2):
	return min(np.linalg.norm(img - imgref1), np.linalg.norm(img - imgref2))


def cut_img(img, start_height, end_height, start_width, end_width):
	return img[:, start_height:end_height, start_width:end_width]

def deplace_bottom_img(empty_img, original, deplacement):
	for row in range(empty_img.shape[1]):
		if row >= deplacement:
			empty_img[:, row, :] = original[:, row - deplacement, :]
	return empty_img

def deplace_upper_img(empty_img, original, deplacement):
	for row in range(empty_img.shape[1]):
		if row <= deplacement:
			empty_img[:, row, :] = original[:, row + deplacement, :]
	return empty_img

def deplace_right_img(empty_img, original, deplacement):
	for col in range(empty_img.shape[2]):
		if col >= deplacement:
			empty_img[:, :, col] = original[:, :, col - deplacement]
	return empty_img

def deplace_left_img(empty_img, original, deplacement):
	for col in range(empty_img.shape[2]):
		if col <= deplacement:
			empty_img[:, :, col] = original[:, :, col + deplacement]
	return empty_img

dataset_name = 'datasets/03_mnist_constant_image_random_row'
X_labeled = np.load(join(dataset_name, "X_labeled.npy"))
y_labeled = np.load(join(dataset_name, "y_labeled.npy"))
X_unlabeled = np.load(join(dataset_name, "X_unlabeled.npy"))
X_val = np.load(join(dataset_name, "X_val.npy"))

# Adapt classes range
y_labeled_adapted = y_labeled.copy()
y_labeled_adapted[y_labeled == 3] = 1

# Cut reference image
three_ref = np.zeros((1, 1, 28, 28))
three_ref = cut_img(X_labeled[1167], 0, 28, 0, 28)
zero_ref = np.zeros((1, 1, 28, 28))
zero_ref = cut_img(X_labeled[1087], 0, 28, 0, 28)


# Keep only side of image without ref img
X_lab_nr = np.zeros((X_labeled.shape[0], 1, 28, 28))
for i in range(X_labeled.shape[0]):
	if calculateDistance(X_labeled[i, :, :, :28], three_ref, zero_ref) > calculateDistance(X_labeled[i, :, :, 28:], three_ref, zero_ref):
		X_lab_nr[i] = cut_img(X_labeled[i], 0, 28, 0, 28)
	else :
		X_lab_nr[i] = cut_img(X_labeled[i], 0, 28, 28, 56)

X_unlab_nr = np.zeros((X_unlabeled.shape[0], 1, 28, 28))
for i in range(X_unlabeled.shape[0]):
	if calculateDistance(X_unlabeled[i, :, :, :28], three_ref, zero_ref) > calculateDistance(X_unlabeled[i, :, :, 28:], three_ref, zero_ref):
		X_unlab_nr[i] = cut_img(X_unlabeled[i], 0, 28, 0, 28)
	else :
		X_unlab_nr[i] = cut_img(X_unlabeled[i], 0, 28, 28, 56)

X_val_nr = np.zeros((X_val.shape[0], 1, 28, 28))
for i in range(X_val.shape[0]):
	if calculateDistance(X_val[i, :, :, :28], three_ref, zero_ref) > calculateDistance(X_val[i, :, :, 28:], three_ref, zero_ref):
		X_val_nr[i] = cut_img(X_val[i], 0, 28, 0, 28)
	else :
		X_val_nr[i] = cut_img(X_val[i], 0, 28, 28, 56)

# Test to shift image to the bottom slightly
X_bottom = np.zeros((X_lab_nr.shape[0], 1, 28, 28))
for i in range(X_lab_nr.shape[0]):
	X_bottom[i] = deplace_bottom_img(X_bottom[i], X_lab_nr[i], 7)

X_upper = np.zeros((X_lab_nr.shape[0], 1, 28, 28))
for i in range(X_lab_nr.shape[0]):
	X_upper[i] = deplace_bottom_img(X_upper[i], X_lab_nr[i], 7)

X_left = np.zeros((X_lab_nr.shape[0], 1, 28, 28))
for i in range(X_lab_nr.shape[0]):
	X_left[i] = deplace_left_img(X_left[i], X_lab_nr[i], 7)

X_right = np.zeros((X_lab_nr.shape[0], 1, 28, 28))
for i in range(X_lab_nr.shape[0]):
	X_right[i] = deplace_right_img(X_right[i], X_lab_nr[i], 7)

x_train, x_test, y_train, y_test = train_test_split(X_lab_nr, y_labeled_adapted)
x_train = np.concatenate((x_train, X_bottom[:250], X_upper[250:500], X_left[500:750], X_right[750:1000]), axis=0)
y_train = np.concatenate((y_train, y_labeled_adapted[:250], y_labeled_adapted[250:500], y_labeled_adapted[500:750], y_labeled_adapted[750:1000]), axis=0)

#X_full = X_lab_left
#Y_full = y_labeled_adapted

# Check cut obtained images
#fig, axs = plt.subplots(10, 10, figsize=(50, 50))
#for i in range(10):
#	for j in range(10):
#		rd_ind = random.choice(range(X_lab_nr.shape[0]))
#		axs[i, j].imshow(X_lab_nr[rd_ind, 0], cmap='gray')
#		axs[i, j].axis('off')
#		axs[i, j].set_title(f"Data {rd_ind} label {y_labeled[rd_ind]}", fontsize=7)
#fig.suptitle('Labeled data example')
#plt.show()

#fig, axs = plt.subplots(10, 10, figsize=(50, 50))
#for i in range(10):
#	for j in range(10):
#		rd_ind = random.choice(range(X_unlab_nr.shape[0]))
#		axs[i, j].imshow(X_unlab_nr[rd_ind, 0], cmap='gray')
#		axs[i, j].axis('off')
#		axs[i, j].set_title(f"Data {rd_ind}", fontsize=7)
#fig.suptitle('Unlabeled data example')
#plt.show()

#fig, axs = plt.subplots(10, 10, figsize=(50, 50))
#for i in range(10):
#	for j in range(10):
#		rd_ind = random.choice(range(X_val_nr.shape[0]))
#		axs[i, j].imshow(X_val_nr[rd_ind, 0], cmap='gray', )
#		axs[i, j].axis('off')
#		axs[i, j].set_title(f"Data {rd_ind}", fontsize=7)
#fig.suptitle('Val data example')
#plt.show()

# Split labeled data into train and test
x_train = torch.from_numpy(x_train).to(device).float()
y_train = torch.from_numpy(y_train).to(device)
x_test = torch.from_numpy(x_test).to(device).float()
y_test = torch.from_numpy(y_test).to(device)

# Create model for 2 classes
model = MNIST_model()
train(model, x_train, y_train, x_test, y_test, epochs=20) # TODO modify epochs

# Predict on val and unlabeled data and save
try:
	os.mkdir('./ex03_results')
except:
	pass

x_unlabeled = torch.from_numpy(X_unlab_nr).to(device).float()
y_unlabeled = model.forward(x_unlabeled)
y_unlabeled_numpy = y_unlabeled.detach().numpy()
y_unlabeled_numpy = np.argmax(y_unlabeled_numpy, axis=1)
y_unlabeled_numpy[y_unlabeled_numpy == 1] = 3
df = pd.DataFrame(y_unlabeled_numpy)
df.to_csv('./ex03_results/y_unlabeled.csv', index=False, header=False)

x_val = torch.from_numpy(X_val_nr).to(device).float()
y_val = model.forward(x_val)
y_val_numpy = y_val.detach().numpy()
y_val_numpy = np.argmax(y_val_numpy, axis=1)
y_val_numpy[y_val_numpy == 1] = 3
df = pd.DataFrame(y_val_numpy)
df.to_csv('./ex03_results/y_val.csv', index=False, header=False)

fig, axs = plt.subplots(10, 10, figsize=(50, 50))
for i in range(10):
	for j in range(10):
		rd_ind = random.choice(range(X_val_nr.shape[0]))
		axs[i, j].imshow(X_val[rd_ind, 0], cmap='gray', )
		axs[i, j].axis('off')
		axs[i, j].set_title(f"Data {rd_ind}, label {y_val_numpy[rd_ind]}", fontsize=7, color='red')
fig.suptitle('Val data example')
plt.show()
