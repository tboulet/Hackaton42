import numpy as np
import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class MNIST_model(nn.Module):

	def __init__(self):
		super().__init__()

		self.layer1 = torch.nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding='same'),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding='same'),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(kernel_size=2),
			nn.Dropout(p=0.25),
		)
		self.layer2 = torch.nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(p=0.25),
		)
		self.layer3 = torch.nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.Dropout(p=0.25)
		)
		self.layer4 = torch.nn.Sequential(
			nn.Flatten(),
			nn.Linear(3136, 256),
			nn.ReLU(),
			nn.BatchNorm1d(256),
			nn.Dropout(p=0.25),
		)
		self.layer5 = torch.nn.Sequential(
			nn.Linear(256, 2),
			nn.Softmax(dim=1),
		)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		return out

def evaluate(model, x_test, y_test):
	model.eval()
	with torch.no_grad():
		y_pred = model.forward(x_test)
	correct_test = (torch.argmax(y_pred, axis=1) == y_test).sum().item()
	model.train()
	return correct_test/len(x_test)

def train(model, x_train, y_train, x_test, y_test, epochs=50, batches_size=128):
	nb_batches = int(x_train.shape[0] / batches_size)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.RMSprop(model.parameters(), lr=0.001)
	for epoch in tqdm(range(epochs)):
		correct = 0
		running_loss = 0.0
		for batch in range(nb_batches):
			inputs = x_train[batch*batches_size:(batch+1)*batches_size]
			labels = y_train[batch*batches_size:(batch+1)*batches_size]
			optimizer.zero_grad()
			outputs = model.forward(inputs)
			correct += (torch.argmax(outputs, axis=1) == labels).sum().item()
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		if epoch % 10 == 0:
			print("accuracy train:", correct/x_train.shape[0], "| accuracy val:", evaluate(model, x_test, y_test))
	outputs = model.forward(x_train)
	correct = (torch.argmax(outputs, axis=1) == y_train).sum().item()
	print("Final : accuracy train:", correct/x_train.shape[0], "| accuracy val:", evaluate(model, x_test, y_test))
