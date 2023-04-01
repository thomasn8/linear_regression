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
import numpy as np

# You can store these values when you normalize the dataset like:
# min_values = X.min(axis=0) and max_values = X.max(axis=0) for the X values, and min_values = y.min() and max_values = y.max() for the y values.
def min_max_normalize(dataset):
    dataset = np.array(dataset)
    # Get the minimum and maximum values for each column
    min_values = dataset.min(axis=0)
    max_values = dataset.max(axis=0)
    # Normalize each column using min-max normalization
    normalized_dataset = (dataset - min_values) / (max_values - min_values)
    return normalized_dataset, min_values, max_values

# Note that you will need to pass in the original minimum and maximum values for the X and y columns
def denormalize_value(value, min_value, max_value):
    denormalized_value = value * (max_value - min_value) + min_value
    return denormalized_value

# Normalization min-max scaling: scales the values to be between 0 and 1
# def min_max_normalization(original_dataset):
# 	dataset = copy.deepcopy(original_dataset)
	
# 	# Calculate the mean and standard deviation of the first column (mileage)
# 	mileage_sum = 0
# 	mileage_count = 0
# 	for i in range(len(dataset)):
# 		mileage_sum += dataset[i][0]
# 		mileage_count += 1
# 	mileage_mean = mileage_sum / mileage_count

# 	mileage_sq_diff_sum = 0
# 	for i in range(len(dataset)):
# 		mileage_sq_diff_sum += (dataset[i][0] - mileage_mean) ** 2
# 	mileage_std = (mileage_sq_diff_sum / (mileage_count - 1)) ** 0.5

# 	# Normalize the first column (mileage)
# 	for i in range(len(dataset)):
# 		dataset[i][0] = (dataset[i][0] - mileage_mean) / mileage_std

# 	# Calculate the minimum and maximum values of each column
# 	min_values = [float('inf')] * len(dataset[0])
# 	max_values = [float('-inf')] * len(dataset[0])
# 	for i in range(len(dataset)):
# 		for j in range(len(dataset[0])):
# 			if dataset[i][j] < min_values[j]:
# 				min_values[j] = dataset[i][j]
# 			if dataset[i][j] > max_values[j]:
# 				max_values[j] = dataset[i][j]

# 	# Normalize each column using min-max scaling
# 	for i in range(len(dataset)):
# 		for j in range(len(dataset[0])):
# 			dataset[i][j] = (dataset[i][j] - min_values[j]) / (max_values[j] - min_values[j])

# 	return dataset
