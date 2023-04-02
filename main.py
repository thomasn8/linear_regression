from LinearRegression import LinearRegression

def main():
	reg = LinearRegression()
	w, b = reg.fit('datas/data.csv', lr=0.01, iter=10000, visualize=False)
	
	print()
	mileage = 0
	reg.predict(mileage, w, b)
	mileage = 30000
	reg.predict(mileage, w, b)
	mileage = -30000
	reg.predict(mileage, w, b)
	mileage = 300000
	reg.predict(mileage, w, b)
	mileage = 999999
	reg.predict(mileage, w, b)

	# mileage = 55555
	# reg.predictAndShow(mileage, w, b)


if __name__ == "__main__":
	main()