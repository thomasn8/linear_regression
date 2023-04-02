import csv, copy, math
import matplotlib.pyplot as plt

class LinearRegression:

	def __init__(self):
		self.lr = None
		self.iter = None
		self.visualize = None
		self.csv_file = None
		self.points = None
		self.X = None
		self.y = None
		self.n = None
		self.b = None
		self.w = None
		self.trained = False


	def readCsvFile(self):
		with open(self.csv_file, 'r') as f:
			reader = csv.reader(f)
			next(reader)  # skip header
			dataset = [[float(row[0]), float(row[1])] for row in reader]
		if len(dataset) < 1:
			raise Exception("csv_file error")
		else:
			return dataset


	def parseDatas(self, csv_file, lr, iter, visualize):
		self.csv_file = csv_file
		self.lr = lr
		self.iter = iter
		self.visualize = visualize
		self.points = self.readCsvFile()
		self.n = len(self.points)
		self.X = []
		self.y = []
		for row in self.points:
			self.X.append(row[0])
			self.y.append(row[1])


	def calculateScaleValue(self, X):
		max = None
		for i in range(len(X)):
			if i == 0:
				max = X[i]
			if X[i] > max:
				max = X[i]
		
		if max:
			log10 = math.floor(math.log(max, 10))
			return 1 * 10 **log10
		return None


	def simpleScaling(self, X, scaler):
		for i in range(len(X)):
			X[i] /= scaler;
		return X


	def scaleDatas(self):
		X = copy.deepcopy(self.X)
		y = copy.deepcopy(self.y)
		divider = self.calculateScaleValue(X)
		X = self.simpleScaling(X, divider)
		return X, y, divider
	

	def visualization(self):
		if self.visualize == True: 
			Xmin = 0
			Xmax = int(max(self.X))
			# take all the points of X-y
			plt.scatter(self.X, self.y, color="black", marker = "o", s = 30)
			# make the line with respect to w and b learned
			plt.plot(list(range(Xmin, Xmax)), [(self.w * x) + self.b for x in range(Xmin, Xmax)], color="red")
			plt.show()


	def predict(self, x, w, b, precision=2):
		if not self.trained:
			raise Exception("train model before using it")
		y_pred = w*x + b
		print('Predicted value for', x, ':', round(y_pred, precision))
		return y_pred


	def predictAndShow(self, x, w, b, precision=2):
		if not self.trained:
			raise Exception("train model before using it")
		
		y_pred = w*x + b
		print(f'Predicted value for {x}: {round(y_pred, precision)}. You can see it on the graph.')

		# take all the learned points
		plt.scatter(self.X, self.y, color="black", marker = "o", s = 30)
		
		# take the predicted point
		plt.scatter(x, y_pred, color="blue", marker = "o", s = 30)

		# define the graph limit for the X axis
		Xmin = 0
		Xmax = int(max(self.X))
		if x > Xmax:
			Xmax = x

		# make the line with respect to w and b learned
		plt.plot(list(range(Xmin, Xmax)), [(self.w * x) + self.b for x in range(Xmin, Xmax)], color="red")
		plt.show()

		return y_pred
		

	def gradientDescent(self, X, y, divider):
		n = self.n
		lr = self.lr
		iter = self.iter

		y_pred = [0.] * n
		w = 0.
		b = 0.
		for _ in range(iter):

			## reset the gradients
			dw = 0. 
			db = 0.

			for i in range(n):
				## prediction
				x = X[i]
				wx = x * w
				y_pred[i] = wx + b

				## error (gradient)
				error = y[i] - y_pred[i]
				dw += -(2/n) * X[i] * error
				db += -(2/n) * error

			## update bias and weights
			w = w - (lr * dw)
			b = b - (lr * db)

			if _ % (iter/10) == 0:
				if _ == 0:
					print("TRAINING THE MODEL:")
				print(f"	Iter. {_}: bias= {b}, weight= {w/divider}")
		
		return (w/divider), b


	def fit(self, csv_file, lr=0.001, iter=10000, visualize=False):
		self.parseDatas(csv_file, lr, iter, visualize)
		X, y, divider = self.scaleDatas()
		self.w, self.b = self.gradientDescent(X, y, divider)
		self.trained = True
		print(f'MODEL TRAINED\n	Results: bias = {self.b}, weight = {self.w}')
		self.visualization()
		return self.w, self.b


## calculate the loss (the mean squared errors) manually 
# def loss_function(b, w, points):
# 	y_pred = [0.] * len(points)		# a list of n points to put the predicted values for each point

# 	j = 0							# index of the raw (or point) we are in
# 	for point in points:
# 		sum_error = 0
# 		yj = point[-1] # actual y value
# 		x = point[:-1] # all the x values
# 		wx = 0
# 		for i in range(len(x)):				# for each x
# 			wx += x[i] * w[i]				# dot products (produit scalaire) of each weight with the corrsponding x value
# 		y_pred[j] = wx + b					# get the predicted_value in list corresponding to each point
# 		error = yj - y_pred[j]				# error = actual_value - predicted_value
# 		sum_error += (error)**2				# add to sum of the squared errors
# 		j+=1
	
# 	mean_error = sum_error/float(len(points))	# divide the sum by the num of point to get the mean squared error
# 	return mean_error