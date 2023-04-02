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
			plt.scatter(self.X, self.y, color="black", marker = "o", s = 30)
			plt.plot(list(range(Xmin, Xmax)), [(self.w * x) + self.b for x in range(Xmin, Xmax)], color="red")
			plt.show()


	def predict(self, x, w, b):
		if not self.trained:
			raise Exception("train model before using it")
		y = w*x + b
		print('Predicted value:', y)
		return w*x + b
	

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
				print(f"	Iter {_}: bias={b}, weight={w/divider}")
		
		return (w/divider), b


	def fit(self, csv_file, lr=0.001, iter=10000, visualize=False):
		self.parseDatas(csv_file, lr, iter, visualize)
		X, y, divider = self.scaleDatas()
		self.w, self.b = self.gradientDescent(X, y, divider)
		self.trained = True
		self.visualization()
		print(f'bias = {self.b}, weight = {self.w}')
		return self.w, self.b
