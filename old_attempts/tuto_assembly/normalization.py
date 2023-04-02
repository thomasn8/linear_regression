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



def denormalization(original_dataset, theta0, theta1):
	mileage_sum = 0
	mileage_count = 0
	price_sum = 0
	price_count = 0
	for i in range(len(original_dataset)):
		mileage_sum += original_dataset[i][0]
		mileage_count += 1
		price_sum += original_dataset[i][1]
		price_count += 1
	mileage_mean = mileage_sum / mileage_count
	price_mean = price_sum / price_count

	mileage_sq_diff_sum = 0
	price_sq_diff_sum = 0
	for i in range(len(original_dataset)):
		mileage_sq_diff_sum += (original_dataset[i][0] - mileage_mean) ** 2
		price_sq_diff_sum += (original_dataset[i][1] - price_mean) ** 2
	mileage_std = (mileage_sq_diff_sum / (mileage_count - 1)) ** 0.5
	price_std = (price_sq_diff_sum / (price_count - 1)) ** 0.5

	min_mileage = min(row[0] for row in original_dataset)
	max_mileage = max(row[0] for row in original_dataset)
	min_price = min(row[1] for row in original_dataset)
	max_price = max(row[1] for row in original_dataset)

	theta0_denormalized = (theta0 * (max_price - min_price) / price_std) - \
						(theta1 * (max_mileage * (max_price - min_price) / price_std)) + \
						(min_price - (theta0 * min_mileage / mileage_std) + \
						(theta1 * min_mileage * (max_price - min_price) / (price_std * max_mileage)))

	theta1_denormalized = theta1 * (max_price - min_price) / (max_mileage * price_std)

	return theta0_denormalized, theta1_denormalized
