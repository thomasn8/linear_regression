from LinearRegression import LinearRegression

## visualization can be set in the fit method and by using the predictAndShow method
def main():

	## dataset 1
	# reg = LinearRegression()
	# reg.fit('datas/data.csv', lr=0.01, it=10000, visualize=False)
	# print()
	# mileage = 30000
	# reg.predict(mileage)
	# reg.predictVisualize(mileage)

	## dataset 2
	reg = LinearRegression()
	reg.fit('datas/cars.csv', lr=0.01, it=10000, visualize=False)
	print()
	years = 0
	mileage = 30000
	reg.predict(years, mileage)
	years = 1
	mileage = 30000
	reg.predict(years, mileage)
	years = 15
	mileage = 30000
	reg.predict(years, mileage)


if __name__ == "__main__":
	main()
