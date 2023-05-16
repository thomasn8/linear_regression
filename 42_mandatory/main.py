from LinearRegression import LinearRegression

## visualizations can be set in the fit method and by using the estimatePriceVisualize method
def main():
	reg = LinearRegression()
	reg.fit('datas/data.csv', learningRate=0.01, iter=10000, visualize=True)
	print()
	mileage = 55555
	reg.estimatePriceVisualize(mileage)
	mileage = 0
	reg.estimatePrice(mileage)
	mileage = 30000
	reg.estimatePrice(mileage)
	mileage = -30000
	reg.estimatePrice(mileage)
	mileage = 300000
	reg.estimatePrice(mileage)
	mileage = 999999
	reg.estimatePrice(mileage)


if __name__ == "__main__":
	main()
