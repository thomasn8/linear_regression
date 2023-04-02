## TO DO
## 3 Optimiser le code et la partie parsing
## 4 Faire passer la partie mandatory
## 5 Adapter pour que l'algo fonctionne avec des datas complexes, > 1 weight


import csv
import copy
from utils import *
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


def gradient_descent(csv_file, lr=0.001, iter=1500, visualize=False):
	points, X, y, y_pred, n = parse_csv_file(csv_file)
	
	## SCALE X VALUES
	X_original = copy.deepcopy(X)
	y_original = copy.deepcopy(y)
	divider = calculate_scale_value(X)
	X = simple_scaling(X, divider)

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
			print(f"Iteration {_}: bias={b}, weight={w/divider}")

	## VISUALIZATION
	if visualize == True: 
		Xmin = 0
		# Xmin = int(min(X_original))
		Xmax = int(max(X_original))
		plt.scatter(X_original, y_original, color="black", marker = "o", s = 30)
		plt.plot(list(range(Xmin, Xmax)), [(w/divider) * x + b for x in range(Xmin, Xmax)], color="red")
		plt.show()

	return b, (w/divider)


## TESTS
visu = True

file = 'datas/student.csv'
b, w = gradient_descent(file, lr=0.0001, iter=1000, visualize=visu)
print(f'{file}: ', 'bias = ', b, 'weight = ', w, '\n')

file = 'datas/data.csv'
b, w = gradient_descent(file, lr=0.01, iter=10000, visualize=visu)
print(f'{file}: ', 'bias = ', b, 'weight = ', w, '\n')
