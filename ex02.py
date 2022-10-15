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

dataset_name = 'datasets/02_mnist_constant_image'
X_labeled = np.load(join(dataset_name, "X_labeled.npy"))
y_labeled = np.load(join(dataset_name, "y_labeled.npy"))
X_unlabeled = np.load(join(dataset_name, "X_unlabeled.npy"))
X_val = np.load(join(dataset_name, "X_val.npy"))

y_labeled_adapted = y_labeled.copy()
y_labeled_adapted[y_labeled == 2] = 1

# Keep only left side of image for
X_lab_left = np.zeros((X_labeled.shape[0], 1, 28, 28))
for i in range(X_labeled.shape[0]):
	X_lab_left[i] = cut_img(X_labeled[i], 0, 28, 0, 28)

X_unlab_left = np.zeros((X_unlabeled.shape[0], 1, 28, 28))
for i in range(X_unlabeled.shape[0]):
	X_unlab_left[i] = cut_img(X_unlabeled[i], 0, 28, 0, 28)

X_val_left = np.zeros((X_val.shape[0], 1, 28, 28))
for i in range(X_val.shape[0]):
	X_val_left[i] = cut_img(X_val[i], 0, 28, 0, 28)

# Test to shift image to the bottom slightly
X_bottom = np.zeros((X_labeled.shape[0], 1, 28, 28))
for i in range(X_labeled.shape[0]):
	X_bottom[i] = deplace_bottom_img(X_bottom[i], X_lab_left[i], 7)

X_upper = np.zeros((X_labeled.shape[0], 1, 28, 28))
for i in range(X_labeled.shape[0]):
	X_upper[i] = deplace_bottom_img(X_upper[i], X_lab_left[i], 7)

X_full = np.concatenate((X_lab_left, X_bottom[:500], X_upper[500:1000]), axis=0)
Y_full = np.concatenate((y_labeled_adapted, y_labeled_adapted[:500], y_labeled_adapted[500:1000]), axis=0)

# Check cut obtained images
#fig, axs = plt.subplots(10, 10, figsize=(50, 50))
#for i in range(10):
#	for j in range(10):
#		rd_ind = random.choice(range(X_lab_left.shape[0]))
#		axs[i, j].imshow(X_lab_left[rd_ind, 0], cmap='gray')
#		axs[i, j].axis('off')
#		axs[i, j].set_title(f"Data {rd_ind} label {y_labeled[rd_ind]}", fontsize=7)
#fig.suptitle('Labeled data example')
#plt.show()

#fig, axs = plt.subplots(10, 10, figsize=(50, 50))
#for i in range(10):
#	for j in range(10):
#		rd_ind = random.choice(range(X_unlab_left.shape[0]))
#		axs[i, j].imshow(X_unlab_left[rd_ind, 0], cmap='gray')
#		axs[i, j].axis('off')
#		axs[i, j].set_title(f"Data {rd_ind}", fontsize=7)
#fig.suptitle('Unlabeled data example')
#plt.show()

#fig, axs = plt.subplots(10, 10, figsize=(50, 50))
#for i in range(10):
#	for j in range(10):
#		rd_ind = random.choice(range(X_val_left.shape[0]))
#		axs[i, j].imshow(X_val_left[rd_ind, 0], cmap='gray', )
#		axs[i, j].axis('off')
#		axs[i, j].set_title(f"Data {rd_ind}", fontsize=7)
#fig.suptitle('Val data example')
#plt.show()

#fig, axs = plt.subplots(10, 10, figsize=(50, 50))
#for i in range(10):
#	for j in range(10):
#		rd_ind = random.choice(range(X_val_left.shape[0]))
#		axs[i, j].imshow(X_more[rd_ind, 0], cmap='gray', )
#		axs[i, j].axis('off')
#		axs[i, j].set_title(f"Data {rd_ind}", fontsize=7)
#fig.suptitle('Val data example')
#plt.show()

# Split labeled data into train and test
x_train, x_test, y_train, y_test = train_test_split(X_full, Y_full)
x_train = torch.from_numpy(x_train).to(device).float()
y_train = torch.from_numpy(y_train).to(device)
x_test = torch.from_numpy(x_test).to(device).float()
y_test = torch.from_numpy(y_test).to(device)

# Create model for 2 classes
model = MNIST_model()
train(model, x_train, y_train, x_test, y_test, epochs=20) # TODO modify epochs

# Predict on val and unlabeled data and save
try:
	os.mkdir('./ex02_results')
except:
	pass

x_unlabeled = torch.from_numpy(X_unlab_left).to(device).float()
y_unlabeled = model.forward(x_unlabeled)
y_unlabeled_numpy = y_unlabeled.detach().numpy()
y_unlabeled_numpy = np.argmax(y_unlabeled_numpy, axis=1)
y_unlabeled_numpy[y_unlabeled_numpy == 1] = 2
df = pd.DataFrame(y_unlabeled_numpy)
df.to_csv('./ex02_results/y_unlabeled.csv', index=False, header=False)

fig, axs = plt.subplots(10, 10, figsize=(50, 50))
for i in range(10):
	for j in range(10):
		rd_ind = random.choice(range(X_val_left.shape[0]))
		axs[i, j].imshow(X_val[rd_ind, 0], cmap='gray', )
		axs[i, j].axis('off')
		axs[i, j].set_title(f"Data {rd_ind}, label {y_unlabeled_numpy[rd_ind]}", fontsize=7, color='red')
fig.suptitle('Unlabeled data example')
plt.show()


x_val = torch.from_numpy(X_val_left).to(device).float()
y_val = model.forward(x_val)
y_val_numpy = y_val.detach().numpy()
y_val_numpy = np.argmax(y_val_numpy, axis=1)
y_val_numpy[y_val_numpy == 1] = 2
df = pd.DataFrame(y_val_numpy)
df.to_csv('./ex02_results/y_val.csv', index=False, header=False)

fig, axs = plt.subplots(10, 10, figsize=(50, 50))
for i in range(10):
	for j in range(10):
		rd_ind = random.choice(range(X_val_left.shape[0]))
		axs[i, j].imshow(X_val[rd_ind, 0], cmap='gray', )
		axs[i, j].axis('off')
		axs[i, j].set_title(f"Data {rd_ind}, label {y_val_numpy[rd_ind]}", fontsize=7, color='red')
fig.suptitle('Val data example')
plt.show()
