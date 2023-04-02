import csv, copy, math
import matplotlib.pyplot as plt

class LinearRegression:

	def __init__(self):
		self.learningRate = None
		self.iter = None
		self.visualize = None
		self.csv_file = None
		self.cars = None
		self.mileages = None
		self.prices = None
		self.m = None
		self.tmp_theta0 = 0.
		self.tmp_theta1 = 0.
		self.theta0 = 0.
		self.theta1 = 0.
		self.trained = False
		self.precision = None


	def readCsvFile(self):
		with open(self.csv_file, 'r') as f:
			reader = csv.reader(f)
			next(reader)  # skip header
			dataset = [[float(row[0]), float(row[1])] for row in reader]
		if len(dataset) < 1:
			raise Exception("csv_file error")
		else:
			return dataset


	def parseDatas(self, csv_file, learningRate, iter, visualize):
		self.csv_file = csv_file
		self.learningRate = learningRate
		self.iter = iter
		self.visualize = visualize
		self.cars = self.readCsvFile()
		self.m = len(self.cars)
		self.mileages = []
		self.prices = []
		for car in self.cars:
			self.mileages.append(car[0])
			self.prices.append(car[1])


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
		X = copy.deepcopy(self.mileages)
		y = copy.deepcopy(self.prices)
		divider = self.calculateScaleValue(X)
		X = self.simpleScaling(X, divider)
		return X, y, divider
	

	def visualization(self):
		if self.visualize == True: 
			Xmin = 0
			Xmax = int(max(self.mileages))
			# take all the points of X-y
			plt.scatter(self.mileages, self.prices, color="blue", marker = "o", s = 30)
			# make the line with respect to w and b learned
			plt.plot(list(range(Xmin, Xmax)), [(self.theta1 * x) + self.theta0 for x in range(Xmin, Xmax)], color="black")
			plt.show()


	def estimate(self, mileage):
		return self.tmp_theta1*mileage + self.tmp_theta0

	def estimatePrice(self, mileage, decimal=2):
		if not self.trained:
			raise Exception("train model before using it")
		y_pred = self.theta1*mileage + self.theta0
		print('Predicted value for', mileage, ':', round(y_pred, decimal))
		return y_pred


	def estimatePriceVisualize(self, mileage, decimal=2):
		if not self.trained:
			raise Exception("train model before using it")
		
		y_pred = self.theta1*mileage + self.theta0
		print(f'Predicted value for {mileage}: {round(y_pred, decimal)}. You can see it on the graph.')

		# take all the learned points
		plt.scatter(self.mileages, self.prices, color="blue", marker = "o", s = 30)
		
		# take the predicted point
		plt.scatter(mileage, y_pred, color="red", marker = "o", s = 30)

		# define the graph limit for the X axis
		Xmin = 0
		Xmax = int(max(self.mileages))
		if mileage > Xmax:
			Xmax = mileage

		# make the line with respect to w and b learned
		plt.plot(list(range(Xmin, Xmax)), [(self.theta1 * x) + self.theta0 for x in range(Xmin, Xmax)], color="black")
		plt.show()

		return y_pred


	## Cost function that minimizes progressively the measures of the divergence between her predictions and the actual values
	def gradientDescent(self, X, y, divider):
		m = self.m
		learningRate = self.learningRate
		iter = self.iter

		for _ in range(iter):

			## for each point in the dataset
			dw = 0.
			db = 0.
			for i in range(m):
				## try to predict the y value based on the learned (at the actual iteration) corresponding weight(s) and overall bias
				y_pred = self.estimate(X[i])

				## calculate the gradients (or derivatives):
				## error = diff between the known value and the predicted value
				## sum all the errors to get the w and b gradients
				error = y_pred - y[i]
				db += error
				dw += error * X[i]

			## update bias and weights substracting the mean_error (= sum_derivatives/n_values) multiplied by the learning rate
			## the learning rate helps to move slowly in the right direction
			## should progressively reach a plateau (if the learning is well calibrated with the num of iteration)
			self.tmp_theta0 -= learningRate * (1/m) * db
			self.tmp_theta1 -= learningRate * (1/m) * dw

			if _ % (iter/10) == 0:
				if _ == 0:
					print("TRAINING THE MODEL:")
				print(f"	Iter. {_}: theta0= {self.tmp_theta0}, theta1= {self.tmp_theta1/divider}")
		
		self.theta0 = self.tmp_theta0
		self.theta1 = self.tmp_theta1/divider
		self.trained = True
		print(f'MODEL TRAINED\n	Results: theta0 = {self.theta0}, theta1 = {self.theta1}')


	## Evaluate the accuracy of the model by measuring the goodness of fit determined by the variance
	def meanSquaredError(self):
		sum_error = 0
		for i in range(self.m):
			y_pred = self.mileages[i] * self.theta1 + self.theta0
			error = y_pred - self.prices[i]
			sum_error += (error)**2
		mean_error = sum_error/self.m
		self.precision = mean_error
		print(f'	Precision: MSE = {round(self.precision, 2)}')


	def fit(self, csv_file, learningRate=0.001, iter=10000, visualize=False):
		self.parseDatas(csv_file, learningRate, iter, visualize)
		X, y, divider = self.scaleDatas()
		self.gradientDescent(X, y, divider)
		self.meanSquaredError()
		self.visualization()
