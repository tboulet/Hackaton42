import numpy as np
import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MNIST_model():
	def __init__(self, nbr_classes):
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
			nn.Linear(64*7*14, 256),
			nn.ReLU(),
			nn.BatchNorm1d(256),
			nn.Dropout(p=0.25),
		),
		self.layer5 = torch.nn.Sequential(
			nn.Linear(256, nbr_classes),
			nn.Softmax(),
		)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		return out

	def train(self, x,y, epochs):
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.RMSprop(self.parameters(), lr=0.001)
		running_loss = 0.0
		for epoch in range(epochs):
			inputs, labels = x, y
			optimizer.zero_grad()
			outputs = self.forward()
			loss = nn.CrossEntropyLoss(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()


