from normalization import min_max_normalization, denormalization

class LinearRegression:

	def __init__(self, lr=0.001, n_iters=1500):
		self.lr = lr
		self.n_iters = n_iters
		self.weights = None
		self.bias = None

	def predict(self, b, w, x):
		return b + (w * x)

	def train(self, dataset):
		datas = min_max_normalization(dataset)
		######################## a ameliorer, fonctionne avec seulement 1 x
		X = []
		y = []
		for row in datas:
			X.append(row[0])
			y.append(row[1])
		X = [X]
		print(datas, '\n')
		print(X, '\n')
		print(y, '\n')
		n_features = len(X)
		# ##########################################################
		
		n_samples = len(y)
		self.weights = [0.] * n_features
		self.bias = 0.
		print(self.weights, '\n')
		print(n_features, '\n')
		print(n_samples, '\n')

		for _ in range(self.n_iters):
			# print('\nITER N*',_)

			# ESTIMATIONS
			y_pred = []
			for j in range(n_samples):
				sum_raw = 0
				feature = 0
				# print('raw x:',datas[j][:-1])
				for i in range(n_features):
					# print(datas[j][i], " * ", self.weights[i])

					# x * weight
					feature = datas[j][i] * self.weights[i]

					# sum of independent variables ponderated by theire weight
					sum_raw += feature
				
				# add the bias and we got a prediction for each raw (each datapoint)
				y_pred.append(sum_raw + self.bias)
				# print(f'y_pred {j}:',y_pred[j], '\n')

			# CALCULATE ERROR
			dw = [0.] * n_features
			db = 0.
			for i in range(n_samples):
				for j in range(n_features):
					# print(dw[j], ' += ', (1 / n_samples), ' * (', y_pred[i], '-', y[i], ') * ', datas[i][j])
					dw[j] += (1 / n_samples) * (y_pred[i] - y[i]) * datas[i][j]
				# print(db, ' += ', (1 / n_samples), ' * (', y_pred[i], '-', y[i], ')')
				db += (1 / n_samples) * (y_pred[i] - y[i])
				# print('\n')

			# UPDATE THE WEIGHTS AND BIAS
			# print(dw)
			# print(db)
			# print('\n')
			for i in range(n_features):
				# print(self.weights[i] , ' -= ', self.lr, ' * ', dw[i])
				self.weights[i] -= self.lr * dw[i]
			# print(self.bias , ' -= ', self.lr, ' * ', db)
			self.bias -= self.lr * db

			if _ % (self.n_iters/10) == 0:
				print(f"Iteration {_}: bias={self.bias}, weight={self.weights}")
		
		print("bias =", self.bias)
		print("weight =", self.weights)
		return denormalization(dataset, self.bias, self.weights[0])	