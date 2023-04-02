import numpy as np
 
class LinearRegression:

	# PARAMETERS
	# def __init__(self, lr = 0.001, n_iters = 1000):
	def __init__(self, lr = 0.001, n_iters = 25):
		self.lr = lr
		self.n_iters = n_iters
		self.weights = None
		self.bias = None
	

	# TRAINING
	def fit(self, X, y):
		
		# X = np.array(X)
		# y = np.array(y)
		# n_samples = len(X)
		# n_features = 1

		print(X)
		print(type(X))
		print(y)
		print(type(y))

		n_samples, n_features = X.shape			# returns a tuple (number of rows, number of columns) containing the dimensions of the array X
		self.weights = np.zeros(n_features)		# create an array of n_features floats initialized to 0. representing the initial weights
		self.bias = 0

		print('n_samples', n_samples)
		print('n_features', n_features)
		print('weights',self.weights)
		print('start:\n')

		# gradient descent
		for i in range(self.n_iters):
			
			# print('time')

			# predict a value using the data set and the current weights/bias
			print('weights',self.weights)
			print('bias',self.bias)
			print('\n')
			y_pred = np.dot(X, self.weights) + self.bias
			
			# compare the predicted value and the actual value with respect to the cost function
			# divide by the number of data point to calculate the gradients
			X1d = X.reshape(-1)
			dw = (1/n_samples) * np.dot(X1d, (y_pred - y))
			db = (1/n_samples) * np.sum(y_pred - y)

			# update the weights and the bias using the learning rate
			self.weights = self.weights - self.lr * dw
			self.bias = self.bias - self.lr * db

			# repeat n_iters times
		
		return self.bias, self.weights


	# PREDICTION
	def predict(self, X):
		X = np.array(X)
		y_pred = np.dot(X, self.weights) + self.bias
		return y_pred



# LEXIC
# x						independent var
# X						all the independent var
# y						dependent var or value to predict (actual value)
# w						weight of an independent var in the prediction
# n						number of data points (samples) in our data set

# lr					learning rate, tell us how fast or slow to search in the direction that the gradient descent tells us to go

# n_samples				n (number of data points)
# n_features			n (number of independent variables)
# n_iters				number of iteration to repeat the gradient descent

# (yi)			 		actual value of a data point
# ŷ						estimated value of a data point 
# (wxi + b)				estimated value of a data point 
# (yi - (wxi + b))2		error squared

# (yi - (wxi + b))2 + ... + (yi+n - (wxi+n + b))2			sum of the squared errors
# 1/n * (sum of the squared errors)							mean squared error

# 1/n * sum(y_pred[i] - y[i])								db: calculate the derivative (gradients) for the bias of a data point
# 1/n * sum(xi * (y_pred[i] - y[i]))						dw: calculate the derivative (gradients) for the weight of a data point