## TO DO: finish the visualization
## IMPROVES: 
# add a visualization 
# add a better standardization?/normalization? 
# add the possibility to ponderate the features-weights

import csv, copy, math
import matplotlib.pyplot as plt

class LinearRegression:

	def __init__(self):
		self.lr = None
		self.it = None
		self.visualize = None
		self.csv_file = None
		self.points = None
		self.X = None
		self.n_x = None
		self.y = None
		self.y_pred = None
		self.errors = None
		self.n = None
		self.b = None
		self.w = None
		self.trained = False
		self.mse = None


	def readCsvFile(self):
		with open(self.csv_file, 'r') as f:
			reader = csv.reader(f)
			next(reader)  # skip header
			
			dataset = []	# all points [ [[x], [x], ... [x], [y]], ... ]
			X = []				# X lists [ [x1], [x2], ... [xn] ]
			y = []				# y list [y, ...]
			n = 0
			for row in reader:
				n+=1
				dataset.append([])					# prepare data point lists
				y.append(float(row[-1]))		# add y values to the y list
				for data in row:
					dataset[n-1].append(float(data)) # add data points

				if n == 1:
					n_x = len(row)-1				# number of features (also weights)
					for _ in range(n_x):
						X.append([])					# prepare X lists
					w = [0.] * n_x					# create and init weights
				rawX = row[:-1]
				for x in range(len(rawX)):
					X[x].append(float(rawX[x]))		# add x values to their list grouped by colomn index

		if n == 0:
			raise Exception("csv_file error")
		else:
			y_pred = [0.] * n						# to put the predicted values for each y
			errors = [0.] * n						# to put the diffs between y[i] y_pred[i]
			return dataset, X, y, y_pred, w, n, n_x, errors


	def parseDatas(self, csv_file, lr, it, visualize):
		self.csv_file = csv_file
		self.lr = lr
		self.it = it
		self.visualize = visualize
		self.b = 0.
		self.points, self.X, self.y, self.y_pred, self.w, self.n, self.n_x, self.errors = self.readCsvFile()
		print(self.X)


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
			X[i] /= scaler
		return X


	def scaleDatas(self):
		X = copy.deepcopy(self.X)
		y = copy.deepcopy(self.y)
		dividers = [0.] * self.n_x
		for i in range(self.n_x):
			dividers[i] = self.calculateScaleValue(X[i])
			X[i] = self.simpleScaling(X[i], dividers[i])
		return X, y, dividers


	def predict(self, *args, decimal=2):
		if not self.trained:
			raise Exception("train model before using it")
		
		y_pred = 0
		for i in range(self.n_x):
			y_pred += args[i] * self.w[i]
		y_pred += self.b
		print('Predicted value :', round(y_pred, decimal))
		return y_pred


	def prepareVariables(self):
		return self.n, self.n_x, self.lr, self.it, self.w, self.b, self.y_pred, self.errors


	## Cost function that minimizes progressively the measures of the divergence between her predictions and the actual values
	def gradientDescent(self, X, y, divider):
		n, n_x, lr, it, w, b, y_pred, errors = self.prepareVariables()

		for _ in range(it):

			## for each point in the dataset
			db = 0.
			dw = [0.] * n_x
			for j in range(n):
				## train/predict using learned weights and b
				row_wxi = 0
				for i in range(n_x):
					row_wxi += w[i] * X[i][j] # raw_sum of the x[i] * weight[i]
				y_pred[j] = row_wxi + b

				## calculate the gradients
				errors[j] = y_pred[j] - y[j]
				db += errors[j]
				for i in range(n_x):
					dw[i] += X[i][j] * errors[j]

			## update b and wi
			b -= (lr * (1/n) * db)
			for i in range(n_x):
				w[i] -= (lr * (1/n) * dw[i])

			## display b and wi progression
			if _ % (it/10) == 0:
				if _ == 0:
					print("TRAINING THE MODEL:")
				print(f"Iter. {_}:")
				print(f"	bias = {b},")
				for i in range(n_x):
					print(f"	weight[{i}] = {w[i]/divider[i]},")
		
		## save and display final values
		self.b = b
		for i in range(n_x):
			self.w[i] = w[i]/divider[i]
		self.trained = True
		print(f'\nMODEL TRAINED - Results:')
		print(f'	bias = {self.b},')
		for i in range(n_x):
			print(f'	weight[{i}] = {self.w[i]},')


	def meanSquaredError(self):
		sum_error = 0
		for j in range(self.n):
			row_wxi = 0
			for i in range(self.n_x):
				row_wxi += self.w[i] * self.X[i][j]
			y_pred = row_wxi + self.b
			error = y_pred - self.y[j]
			sum_error += (error)**2
		mean_error = sum_error/self.n
		self.mse = mean_error
		print(f'	Precision: MSE = {round(self.mse, 2)}')


	def fit(self, csv_file, lr=0.001, it=10000, visualize=False):
		self.parseDatas(csv_file, lr, it, visualize)
		X, y, dividers = self.scaleDatas()
		self.gradientDescent(X, y, dividers)
		self.meanSquaredError()
		# self.visualization()
