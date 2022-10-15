import numpy as np
import random

def generate_random_noise(array_img, labels):
	new_array = np.zeros((500, array_img.shape[1], array_img.shape[2], array_img.shape[3]))
	new_labels = np.zeros((500,)).astype(dtype=np.int64)
	var = np.linspace(0.1, 0.8, 6)
	for i in range(500):
		random_idx = random.randint(0, array_img.shape[0])
		random_var = np.random.randint(0, 6)
		new_array[i] = add_noise(var[random_var], array_img[random_idx])
		new_labels[i] = labels[random_idx]
	return new_array, new_labels

def add_noise(var, img):
	i, j, k = img.shape
	mean = 0
	sigma = var**0.5
	gauss = np.random.normal(mean, sigma, (i,j,k))
	gauss = gauss.reshape(i, j, k)
	return gauss
