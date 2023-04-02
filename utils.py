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
import math

def calculate_scale_value(X):
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

def simple_scaling(X, scaler):
	for i in range(len(X)):
		X[i] /= scaler;
	return X

def min_max_normalize(dataset):
    dataset = np.array(dataset)
    # Get the minimum and maximum values for each column
    min_values = dataset.min(axis=0)
    max_values = dataset.max(axis=0)
    # Normalize each column using min-max normalization
    normalized_dataset = (dataset - min_values) / (max_values - min_values)
    return normalized_dataset, min_values, max_values
