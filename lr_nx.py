import matplotlib.pyplot as plt
import csv

## return in this order: 
## all the dataset (= points), lists for x values, grouped by colomn index, 
## list of actual values (= values to predict in the training), empty list to predict values, 
## list of weights, bias, number of points, number of x values in a point (= number of weight)
def parse_csv_file(filename):
	with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter = ",")
		next(reader)  # skip header
		
		dataset = []
		X = []
		y = []
		w = []
		b = 0.
		n = 0
		n_x = 0

		for row in reader:
			n+=1
			dataset.append([])			# add data points (1 raw = 1 data point) to the data set 
			for data in row:
				dataset[n-1].append(float(data))

			if n == 1:
				n_x = len(row)-1
				for _ in range(n_x):
					X.append([])		# create lists for x values, grouped by colomn index
				w = [0.] * n_x			# create and init weights
			rawX = row[:-1]
			for x in range(len(rawX)):
				X[x].append(float(rawX[x]))	# feeds x lists in X list

			y.append(float(row[-1]))	# feeds y list
		
		y_pred = [0.] * n				# a list of n points to put the predicted values for each point

	return dataset, X, y, y_pred, w, b, n, n_x


## calculate the loss (the mean squared errors) manually 
def loss_function(b, w, points):
	y_pred = [0.] * len(points)		# a list of n points to put the predicted values for each point

	j = 0							# index of the raw (or point) we are in
	for point in points:
		sum_error = 0
		yj = point[-1] # actual y value
		x = point[:-1] # all the x values
		wx = 0
		for i in range(len(x)):				# for each x
			wx += x[i] * w[i]				# dot products (produit scalaire) of each weight with the corrsponding x value
		y_pred[j] = wx + b					# get the predicted_value in list corresponding to each point
		error = yj - y_pred[j]				# error = actual_value - predicted_value
		sum_error += (error)**2				# add to sum of the squared errors
		j+=1
	
	mean_error = sum_error/float(len(points))	# divide the sum by the num of point to get the mean squared error
	return mean_error


def gradient_descent(csv_file, lr=0.001, iter=1500):
	points, X, y, y_pred, w, b, n, n_x = parse_csv_file(csv_file)
	dw = [0.] * n_x
	db = 0.

	# visualization
	plt.scatter(X[0], y, color = "b", marker = "o", s = 30)
	plt.show()


	for _ in range(iter):

		## prediction
		j = 0
		for point in points:
			x = point[:-1] # all the x values
			wx = 0.
			for i in range(len(x)):				# for each x
				wx += x[i] * w[i]				# dot products (produit scalaire) of each weight with the corrsponding x value
			y_pred[j] = wx + b					# get the predicted_value in list corresponding to each point
			j+=1


		## error (gradient)
		sum_error1 = 0
		# for i in range(n_x):					# ERROR IN THIS BLOCK (works with one w but not more)
		for i in range(n):
			error = y_pred[i] - y[i]
			xi = points[i][:-1]					# all the x for the corresponding y
			sum_error1 += 2*xi[0] * error		#################################################### CHEAT
		dw[0] = (1/n) * sum_error1				# CHEAT each weight has his own gradient to modify

		sum_error2 = 0
		for i in range(n):
			error = y_pred[i] - y[i]
			sum_error2 += 2 * error
		db = (1/n) * sum_error2					# one bias so one gradient


		## update bias and weights
		for i in range(n_x):
			w[i] = w[i] - (lr * dw[i]) 			# each weight has his own gradient to modify
		b = b - (lr * db)

	return b, w



theta0, theta1 = gradient_descent('youtube.csv', lr=0.0001, iter=300)
print(theta0, theta1)