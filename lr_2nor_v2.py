## TO DO
## 1 Faire fonctionner la denormalization
## 2 Faire marcher la visualization
## 3 Optimiser le code et la partie parsing
## 4 Faire passer la partie mandatory
## 5 Adapter pour que l'algo fonctionne avec des datas complexes, > 1 weight


import csv
import copy
from utils import min_max_normalize, denormalize_value
import matplotlib.pyplot as plt


def parse_csv_file(filename):
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		next(reader)  # skip header
		datas = [[float(row[0]), float(row[1])] for row in reader]

	X = []
	y = []
	n = 0
	for row in datas:
		X.append(row[0])
		y.append(row[1])
		n+=1
		
	y_pred = [0.] * n

	return datas, X, y, y_pred, n


def gradient_descent(csv_file, lr=0.001, iter=1500):
	points, X, y, y_pred, n = parse_csv_file(csv_file)
	

	## Normalization
	# X_original = copy.deepcopy(X)
	# y_original = copy.deepcopy(y)
	# normalized_points, min_values, max_values = min_max_normalize(points) # original min max values : [x_min y_min] [x_max y_max]
	# X = normalized_points[:, 0]
	# y = normalized_points[:, 1]


	w = 0.
	b = 0.
	## START GRANDIENT DESCENT
	for _ in range(iter):

		## reset the gradient
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
			print(f"Iteration {_}: bias={b}, weight={w}")

	return b, w, X, y

	## Denormalization
	# plt.scatter(X, y, color="black", marker = "o", s = 30)
	# plt.plot(list(range(0, 1)), [w * x + b for x in range(0, 1)], color="red") #  youtube.csv
	# plt.show()
	# w_denormalized = denormalize_value(w, min_values[0], max_values[0])
	# b_denormalized = denormalize_value(b, min_values[1], max_values[1])
	# return b_denormalized, w_denormalized, X_original, y_original




## normalization alone 					OK
## gradient descent alone 				OK
## gradient descent + normalization 	ERROR: probably in denormalization

print('\n1st data set: youtube.csv')
file = 'youtube.csv'
b, w, X, y = gradient_descent(file, lr=0.0001, iter=1000)
print(b, w)
plt.scatter(X, y, color="black", marker = "o", s = 30)
plt.plot(list(range(20, 80)), [w * x + b for x in range(20, 80)], color="red") #  youtube.csv
plt.show()






## normalization alone					OK
## gradient descent alone				ERROR: points are too big so we quickly get 'nan' values
## gradient descent + normalization		ERROR:

print('\n2nd data set: data.csv')
file = 'data.csv'
b, w, X, y = gradient_descent(file, lr=0.1, iter=1000)
print(b, w)
plt.scatter(X, y, color="black", marker = "o", s = 30)
plt.plot(list(range(0, 10000)), [w * x + b for x in range(0, 10000)], color="red") # data.csv
plt.show()