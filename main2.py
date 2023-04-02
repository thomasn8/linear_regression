from LinearRegressionPolyWeights import LinearRegression

## visualization can be set in the fit method and by using the predictAndShow method
def main():
	# dataset 0
	reg = LinearRegression()
	reg.fit('datas/cars.csv', lr=0.01, it=10000, visualize=False)
	# print()
	# mileage = 30000
	# reg.predictVisualize(mileage)

	# # dataset 1
	# reg = LinearRegression()
	# reg.fit('datas/data.csv', lr=0.01, it=10000, visualize=False)
	# print()
	# mileage = 30000
	# reg.predictVisualize(mileage)

	# # dataset 2
	# reg = LinearRegression()
	# reg.fit('datas/student.csv', lr=0.01, it=10000, visualize=False)
	# print()
	# hours = 100
	# reg.predictVisualize(hours)


if __name__ == "__main__":
	main()
