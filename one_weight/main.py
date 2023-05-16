from LinearRegression import LinearRegression

## visualizations can be set in the fit method and by using the predictAndShow method
def main():
	reg = LinearRegression()
	reg.fit('datas/data.csv', lr=0.01, iter=10000, visualize=False)
	
	print()

	mileage = 55555
	reg.predictVisualize(mileage)

	mileage = 0
	reg.predict(mileage)
	mileage = 30000
	reg.predict(mileage)
	mileage = -30000
	reg.predict(mileage)
	mileage = 300000
	reg.predict(mileage)
	mileage = 999999
	reg.predict(mileage)


if __name__ == "__main__":
	main()
