import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import random
import argparse
import sys

# NOT WORKING FOR SOME DATASETS (00 for instance)
parser = argparse.ArgumentParser("Show data program")
parser.add_argument("data_dir", type=str, default="data", help="dataset directory")
parser.add_argument("--data_id", type=int, default=0, help="data from dataset at id")
parser.add_argument("--all_label", type=bool, default=False, help="show all data from labelled dataset")
args = parser.parse_args()

X_labeled = np.load(join(args.data_dir, "X_labeled.npy"))
y_labeled = np.load(join(args.data_dir, "y_labeled.npy"))
X_unlabeled = np.load(join(args.data_dir, "X_unlabeled.npy"))
X_val = np.load(join(args.data_dir, "X_val.npy"))

if args.data_id:
	fig, axs = plt.subplots(10, 10, figsize=(50, 50))
	for i in range(10):
		for j in range(10):
			axs[i, j].imshow(X_val[args.data_id, 0], cmap='gray')
			axs[i, j].axis('off')
			axs[i, j].set_title(f"Data {args.data_id}", fontsize=7)
	fig.suptitle('Val data example')
	plt.show()
	sys.exit()

if args.all_label:
	x = 0
	while x < X_labeled.shape[0]:
		fig, axs = plt.subplots(10, 10, figsize=(50, 50))
		for i in range(10):
			for j in range(10):
				axs[i, j].imshow(X_labeled[x, 0], cmap='gray')
				axs[i, j].axis('off')
				axs[i, j].set_title(f"Data {x} label {y_labeled[x]}", fontsize=7, color="red")
				x = x + 1

		fig.suptitle('Labeled data example')
		plt.show()
		print(x)
	sys.exit()

print('X label:', X_labeled.shape)
print('Y label:', y_labeled.shape)
print('X val:', X_val.shape)
print('X unlabeled:', X_unlabeled.shape)

fig, axs = plt.subplots(10, 10, figsize=(50, 50))
for i in range(10):
	for j in range(10):
		rd_ind = random.choice(range(X_labeled.shape[0]))
		axs[i, j].imshow(X_labeled[rd_ind, 0], cmap='gray')
		axs[i, j].axis('off')
		axs[i, j].set_title(f"Data {rd_ind} label {y_labeled[rd_ind]}", fontsize=7, color="red")
fig.suptitle('Labeled data example')
plt.show()

fig, axs = plt.subplots(10, 10, figsize=(50, 50))
for i in range(10):
	for j in range(10):
		rd_ind = random.choice(range(X_unlabeled.shape[0]))
		axs[i, j].imshow(X_unlabeled[rd_ind, 0], cmap='gray')
		axs[i, j].axis('off')
		axs[i, j].set_title(f"Data {rd_ind}", fontsize=7, color="red")
fig.suptitle('Unlabeled data example')
plt.show()

fig, axs = plt.subplots(10, 10, figsize=(50, 50))
for i in range(10):
	for j in range(10):
		rd_ind = random.choice(range(X_val.shape[0]))
		axs[i, j].imshow(X_val[rd_ind, 0], cmap='gray')
		axs[i, j].axis('off')
		axs[i, j].set_title(f"Data {rd_ind}", fontsize=7, color="red")
fig.suptitle('Val data example')
plt.show()
