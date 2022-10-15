import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import argparse

# NOT WORKING FOR SOME DATASETS (00 for instance)
parser = argparse.ArgumentParser("Show data program")
parser.add_argument("data_dir", type=str, default="data", help="dataset directory")
args = parser.parse_args()

X_labeled = np.load(join(args.data_dir, "X_labeled.npy"))
y_labeled = np.load(join(args.data_dir, "y_labeled.npy"))
X_unlabeled = np.load(join(args.data_dir, "X_unlabeled.npy"))
X_val = np.load(join(args.data_dir, "X_val.npy"))

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
		axs[i, j].set_title(y_labeled[rd_ind], fontsize=7)
fig.suptitle('Labeled data example')
plt.show()

fig, axs = plt.subplots(10, 10, figsize=(50, 50))
for i in range(10):
	for j in range(10):
		rd_ind = random.choice(range(X_unlabeled.shape[0]))
		axs[i, j].imshow(X_unlabeled[rd_ind, 0], cmap='gray')
		axs[i, j].axis('off')
fig.suptitle('Unlabeled data example')
plt.show()

fig, axs = plt.subplots(10, 10, figsize=(50, 50))
for i in range(10):
	for j in range(10):
		rd_ind = random.choice(range(X_val.shape[0]))
		axs[i, j].imshow(X_val[rd_ind, 0], cmap='gray', )
		axs[i, j].axis('off')
fig.suptitle('Val data example')
plt.show()
