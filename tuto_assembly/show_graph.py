import matplotlib.pyplot as plt

def show_graph(dataset):
	X = []
	Y = []
	for row in dataset:
		X.append(row[0])
		Y.append(row[1])
	plt.scatter(X, Y, color = "b", marker = "o", s = 30)
	plt.show()
