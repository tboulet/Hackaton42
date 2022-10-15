import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from os.path import join

dataset_name = 'datasets/03_mnist_constant_image_random_row'

df = pd.read_csv('./ex03_results/y_val.csv', header=None)
array = df.to_numpy()
X_val = np.load("datasets/03_mnist_constant_image_random_row/X_val.npy")

fig, axs = plt.subplots(10, 10, figsize=(50, 50))
for i in range(10):
	for j in range(10):
		rd_ind = random.choice(range(array.shape[0]))
		axs[i, j].imshow(X_val[rd_ind, 0], cmap='gray', )
		axs[i, j].axis('off')
		axs[i, j].set_title(f"Data {rd_ind}, label {array[rd_ind]}", fontsize=7, color='red')
fig.suptitle('Val data example')
plt.show()