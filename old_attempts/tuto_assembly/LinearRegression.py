import numpy as np


class LinearRegression:

	def __init__(self, lr = 0.001, n_iters=1500):
		self.lr = lr
		self.n_iters = n_iters
		self.weights = None
		self.bias = None

	def fit(self, X, y):
		print(X, type(X))
		print(y, type(y))
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		# for _ in range(self.n_iters):
		for _ in range(1):
			y_pred = np.dot(X, self.weights) + self.bias

			dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
			print(dw)
			db = (1/n_samples) * np.sum(y_pred-y)

			self.weights = self.weights - self.lr * dw
			self.bias = self.bias - self.lr * db

		return self.bias, self.weights

	def predict(self, X, weights, bias):
		y_pred = np.dot(X, weights) + bias
		return y_pred