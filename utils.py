# EXTRACT DATAS 
import csv

def read_csv_file(filename):
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		next(reader)  # skip header
		data = [[float(row[0]), float(row[1])] for row in reader]
	return data



# VISUALIZATION OF DATAS
import matplotlib.pyplot as plt

def show_graph(dataset):
	X = []
	Y = []
	for row in dataset:
		X.append(row[0])
		Y.append(row[1])
	plt.scatter(X, Y, color = "b", marker = "o", s = 30)
	plt.show()



# NORMALIZE DATAS
import copy

# Normalization min-max scaling: scales the values to be between 0 and 1
def min_max_normalization(original_dataset):
	dataset = copy.deepcopy(original_dataset)
	
	# Calculate the mean and standard deviation of the first column (mileage)
	mileage_sum = 0
	mileage_count = 0
	for i in range(len(dataset)):
		mileage_sum += dataset[i][0]
		mileage_count += 1
	mileage_mean = mileage_sum / mileage_count

	mileage_sq_diff_sum = 0
	for i in range(len(dataset)):
		mileage_sq_diff_sum += (dataset[i][0] - mileage_mean) ** 2
	mileage_std = (mileage_sq_diff_sum / (mileage_count - 1)) ** 0.5

	# Normalize the first column (mileage)
	for i in range(len(dataset)):
		dataset[i][0] = (dataset[i][0] - mileage_mean) / mileage_std

	# Calculate the minimum and maximum values of each column
	min_values = [float('inf')] * len(dataset[0])
	max_values = [float('-inf')] * len(dataset[0])
	for i in range(len(dataset)):
		for j in range(len(dataset[0])):
			if dataset[i][j] < min_values[j]:
				min_values[j] = dataset[i][j]
			if dataset[i][j] > max_values[j]:
				max_values[j] = dataset[i][j]

	# Normalize each column using min-max scaling
	for i in range(len(dataset)):
		for j in range(len(dataset[0])):
			dataset[i][j] = (dataset[i][j] - min_values[j]) / (max_values[j] - min_values[j])

	return dataset
