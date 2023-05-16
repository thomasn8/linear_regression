from LinearRegression import LinearRegression
import csv
from normalization import min_max_normalization, denormalization
from show_graph import show_graph
import numpy as np

def read_csv_file(filename):
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		next(reader)  # skip header
		data = [[float(row[0]), float(row[1])] for row in reader]
	return data


def main():
	dataset = read_csv_file('data.csv')
	# datas = read_csv_file('data.csv')
	# dataset = min_max_normalization(datas)
	X = np.array([])
	y = np.array([])
	for row in dataset:
		X = np.append(X, row[0])
		y = np.append(y, row[1])
	X = [X]
	X = np.array(X)
	X = X.T
	# show_graph(dataset)
	
	reg = LinearRegression(lr=0.001, n_iters=1500)
	theta0, theta1 = reg.fit(X, y)
	

	# theta0, theta1 = denormalization(datas, theta0, theta1)
	mileage = 200000
	price = reg.predict(theta0, theta1, mileage)
	print("\ntheta0 =", theta0)
	print("theta1 =", theta1)
	print(f"Predicted price for mileage {mileage} : {price}")
	dataset.append([mileage, price])

	# show_graph(dataset)

if __name__ == "__main__":
	main()