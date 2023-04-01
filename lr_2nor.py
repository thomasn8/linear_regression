import matplotlib.pyplot as plt
import csv
from utils import min_max_normalization

# 1 Faire fonctionner la denormalization
# 2 Faire marcher la visualization
# 3 Optimiser le code et la partie parsing
# 4 Faire passer la partie mandatory
# 5 Adapter pour que l'algo fonctionne avec des datas complexes, > 1 weight

def parse_csv_file(filename):
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		next(reader)  # skip header
		datas_ori = [[float(row[0]), float(row[1])] for row in reader]

	normalization = False
	for row in datas_ori:
		for value in row:
			if value > 1000:
				normalization = True

	X_ori = []
	y_ori = []
	if normalization == True:
		datas = min_max_normalization(datas_ori)
		for row in datas_ori:
			X_ori.append(row[0])
			y_ori.append(row[1])
	else:
		datas = datas_ori

	X = []
	y = []
	n = 0
	for row in datas:
		X.append(row[0])
		y.append(row[1])
		n+=1
		
	y_pred = [0.] * n

	return normalization, datas_ori, X_ori, y_ori, datas, X, y, y_pred, n


def check_parsing(datas_ori, X_ori, y_ori, points, X, y, y_pred, n):
	## check if parsing worked:
	# print('\npoints:',points)
	# print(type(points[0][0]))
	# print('\nX:',X)
	# print('\ny:',y)
	# print('\ny_pred:',y_pred)
	# print('\nw:',w)
	# print('\nb:',b)
	# print('\nn:',n)
	# print('\nn_x:',n_x)
	# print('\ndw:',dw)
	return


def gradient_descent(csv_file, lr=0.001, iter=1500):
	norma, datas_ori, X_ori, y_ori, points, X, y, y_pred, n = parse_csv_file(csv_file)
	# check_parsing(datas_ori, X_ori, y_ori, points, X, y, y_pred, n)

	w = 0.
	b = 0.
	# START GRANDIENT DESCENT
	for _ in range(iter):

		## prediction
		for i in range(n):
			x = X[i]
			wx = x * w
			y_pred[i] = wx + b

		## error (gradient)
		dw = 0. 
		db = 0.
		for i in range(n):
			error = y[i] - y_pred[i]
			dw += -(2/n) * X[i] * error
			db += -(2/n) * error

		## update bias and weights
		w = w - (lr * dw)
		b = b - (lr * db)

		if _ % (iter/10) == 0:
			print(f"Iteration {_}: bias={b}, weight={w}")

		# DENORMALIZER DATAS ?
		# if norma === True:
			#  denoraliser w et b
			# scatter X_ori y_ori

	## visualization
	# plt.scatter(X, y, color = "b", marker = "o", s = 30)
	plt.scatter(X, y, color="black")
	if lr < 0.001:
		plt.plot(list(range(20, 80)), [w * x + b for x in range(20, 80)], color="red") # draw line with w/b 
	else:
		plt.plot(list(range(0, 2)), [w * x + b for x in range(0, 2)], color="red") # draw line with w/b 
	plt.show()

	return b, w


theta0, theta1 = gradient_descent('youtube.csv', lr=0.0001, iter=1000)
# theta0, theta1 = gradient_descent('data.csv', lr=0.1, iter=1000)
print(theta0, theta1)
