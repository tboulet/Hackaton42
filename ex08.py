import numpy as np
from os.path import join
from NN.model_for_mnist import *
import numpy as np
from sklearn.model_selection import train_test_split
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from scipy import ndimage

use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

class denoising_model(nn.Module):
  def __init__(self):
    super(denoising_model,self).__init__()
    self.encoder=nn.Sequential(
                  nn.Linear(28*28,256),
                  nn.ReLU(True),
                  nn.Linear(256,128),
                  nn.ReLU(True),
                  nn.Linear(128,64),
                  nn.ReLU(True)
        
                  )
    
    self.decoder=nn.Sequential(
                  nn.Linear(64,128),
                  nn.ReLU(True),
                  nn.Linear(128,256),
                  nn.ReLU(True),
                  nn.Linear(256,28*28),
                  nn.Sigmoid(),
                  )
    
 
  def forward(self,x):
    x=self.encoder(x)
    x=self.decoder(x)
    
    return x

def filter_img(array):
	array[array < 0.0] = 0.0
	array[array > 1.0] = 1.7
	return array

def salt_and_pepper(img):
	row , col = img[0].shape
	number_of_pixels = random.randint(50, 200)
	for i in range(number_of_pixels):
		y_coord=random.randint(0, row - 1)
		x_coord=random.randint(0, col - 1)
		img[0, y_coord, x_coord] = 1.7
	img[img < 0.0] = 0.0
	img[img > 1.0] = 1.7
	return img


def find_number(img_part1, img_part2):
	if np.var(img_part1) > np.var(img_part2):
		return img_part1
	return img_part2

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

dataset_name = 'datasets2/08_mnist_sum_noise_level'
X_labeled = np.load(join(dataset_name, "X_labeled.npy"))
y_labeled = np.load(join(dataset_name, "y_labeled.npy"))
X_unlabeled = np.load(join(dataset_name, "X_unlabeled.npy"))
X_val = np.load(join(dataset_name, "X_val.npy"))

# Adapt classes range
y_labeled_adapted = y_labeled.copy()
y_labeled_adapted[y_labeled == 8] = 1

X_bottom = np.zeros((X_labeled.shape[0], 1, 28, 28))
for i in range(X_labeled.shape[0]):
	X_bottom[i] = deplace_bottom_img(X_bottom[i], X_labeled[i], 7)

X_upper = np.zeros((X_labeled.shape[0], 1, 28, 28))
for i in range(X_labeled.shape[0]):
	X_upper[i] = deplace_bottom_img(X_upper[i], X_labeled[i], 7)

X_left = np.zeros((X_labeled.shape[0], 1, 28, 28))
for i in range(X_labeled.shape[0]):
	X_left[i] = deplace_left_img(X_left[i], X_labeled[i], 7)

X_right = np.zeros((X_labeled.shape[0], 1, 28, 28))
for i in range(X_labeled.shape[0]):
	X_right[i] = deplace_right_img(X_right[i], X_labeled[i], 7)

X_labeled_concat = np.concatenate((X_labeled, X_bottom[:250], X_upper[250:500], X_left[500:750], X_right[750:1000]), axis=0)
y_labeled_concat = np.concatenate((y_labeled_adapted, y_labeled_adapted[:250], y_labeled_adapted[250:500], y_labeled_adapted[500:750], y_labeled_adapted[750:1000]), axis=0)


for img in range(X_labeled_concat.shape[0]):
	if (y_labeled_concat[img]) == 1:
		X_labeled_concat[img] = filter_img(X_labeled_concat[img])
		#X_labeled_concat[img] = salt_and_pepper(X_labeled_concat[img])
	else:
		X_labeled_concat[img] = salt_and_pepper(X_labeled_concat[img])
	#X_labeled_concat[img] = ndimage.gaussian_filter(X_labeled_concat[img], 2)
	X_labeled_concat[img] = ndimage.median_filter(X_labeled_concat[img], 3)

for img in range(X_val.shape[0]):
	X_val[img] = salt_and_pepper(X_val[img])
	X_val[img] = filter_img(X_val[img])

for img in range(X_unlabeled.shape[0]):
	X_unlabeled[img] = salt_and_pepper(X_unlabeled[img])
	X_unlabeled[img] = filter_img(X_unlabeled[img])

# Check obtained images
fig, axs = plt.subplots(10, 10, figsize=(50, 50))
for i in range(10):
	for j in range(10):
		rd_ind = random.choice(range(X_labeled_concat.shape[0]))
		axs[i, j].imshow(X_labeled_concat[rd_ind, 0], cmap='gray')
		axs[i, j].axis('off')
		axs[i, j].set_title(f"Data {rd_ind} label {y_labeled_concat[rd_ind]}", fontsize=7)
fig.suptitle('Labeled data example')
plt.show()

exit()

x_train, x_test, y_train, y_test = train_test_split(X_labeled_concat, y_labeled_concat)


# Split labeled data into train and test
x_train = torch.from_numpy(x_train).to(device).float()
y_train = torch.from_numpy(y_train).to(device)
x_test = torch.from_numpy(x_test).to(device).float()
y_test = torch.from_numpy(y_test).to(device)

# Create model for 2 classes
model = MNIST_model() # BE CAREFUL for this one, need to *2 in Linear input size
train(model, x_train, y_train, x_test, y_test, epochs=25) # TODO modify epochs

# Predict on val and unlabeled data and save
try:
	os.mkdir('./ex08_results')
except:
	pass

x_unlabeled = torch.from_numpy(X_unlabeled,).to(device).float()
y_unlabeled = model.forward(x_unlabeled)
y_unlabeled_numpy = y_unlabeled.detach().numpy()
y_unlabeled_numpy = np.argmax(y_unlabeled_numpy, axis=1)
y_unlabeled_numpy[y_unlabeled_numpy == 1] = 8
df = pd.DataFrame(y_unlabeled_numpy)
df.to_csv('./ex08_results/y_unlabeled.csv', index=False, header=False)

x_val = torch.from_numpy(X_val).to(device).float()
y_val = model.forward(x_val)
y_val_numpy = y_val.detach().numpy()
y_val_numpy = np.argmax(y_val_numpy, axis=1)
y_val_numpy[y_val_numpy == 1] = 8
df = pd.DataFrame(y_val_numpy)
df.to_csv('./ex08_results/y_val.csv', index=False, header=False)

fig, axs = plt.subplots(10, 10, figsize=(50, 50))
for i in range(10):
	for j in range(10):
		rd_ind = random.choice(range(X_val.shape[0]))
		axs[i, j].imshow(X_val[rd_ind, 0], cmap='gray', )
		axs[i, j].axis('off')
		axs[i, j].set_title(f"Data {rd_ind}, label {y_val_numpy[rd_ind]}", fontsize=7, color='red')
fig.suptitle('Val data example')
plt.show()