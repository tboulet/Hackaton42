from PIL import Image
import os, sys
import cv2
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import random


'''
Converts all images in a directory to '.npy' format.
Use np.save and np.load to save and load the images.
Use it for training your neural networks in ML/DL projects. 
'''

# Path to image directory
path = "datasets/999_humans/humans/"

def load_dataset(dirct):
	img = []
	files = os.listdir(dirct)
	files.sort()
	# Append images to a list
	for item in files:
		print(item)
		if os.path.isfile(join(dirct, item)):
			im = Image.open(dirct + "/" + item).convert("RGB")
			im = np.array(im)
			img.append(im)
	return img

X_labeled_dir = join(path, "labeled")
X_unlabeled_dir = join(path, "unlabeled")
X_val_dir = join(path, "val")

# Convert and save the list of images in '.npz' format
# If already converted, load numpy array from npz file
if not os.path.isfile(join(path, 'X_labeled.npz')):
	x_lab_img = load_dataset(X_labeled_dir)
	x_lab_npy = np.array(x_lab_img)
	np.savez("datasets/999_humans/humans/X_labeled.npz", x_lab_npy)
else:
	data = np.load(join(path, 'X_labeled.npz'))
	x_lab_npy = data['arr_0']
print(x_lab_npy.shape)

# DO NOT LAUNCH ! FAR TOO BIG ! May need another solution to convert
#if not os.path.isfile(join(path, 'X_unlabeled.npz')):
#	x_unlab_img = load_dataset(X_unlabeled_dir)
#	x_unlab_npy = np.array(x_unlab_img)
#	print(x_unlab_npy.shape)
#	np.savez("datasets/999_humans/humans/X_unlabeled.npz", x_unlab_npy)
#else:
#	x_unlab_npy = np.load(join(path, 'X_unlabeled.npz'))

if not os.path.isfile(join(path, 'X_val.npz')):
	x_val_img = load_dataset(X_val_dir)
	x_val_npy = np.array(x_val_img)
	np.savez("datasets/999_humans/humans/X_val.npz", x_val_npy)
else:
	data = np.load(join(path, 'X_val.npz'))
	x_val_npy = data['arr_0']
print(x_val_npy.shape)

y_labeled = np.load(join(path, "y_labeled.npy"))

# Show the images
fig, axs = plt.subplots(10, 10, figsize=(50, 50))
for i in range(10):
	for j in range(10):
		rd_ind = random.choice(range(x_lab_npy.shape[0]))
		axs[i, j].imshow(x_lab_npy[rd_ind])
		axs[i, j].axis('off')
		axs[i, j].set_title(f"Data {rd_ind}, label {y_labeled[rd_ind]}", fontsize=7, color="red")
fig.suptitle('Labeled data example')
plt.show()

fig, axs = plt.subplots(10, 10, figsize=(50, 50))
for i in range(10):
	for j in range(10):
		rd_ind = random.choice(range(x_val_npy.shape[0]))
		axs[i, j].imshow(x_val_npy[rd_ind])
		axs[i, j].axis('off')
		axs[i, j].set_title(f"Data {rd_ind}", fontsize=7, color="red")
fig.suptitle('Val data example')
plt.show()
