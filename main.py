from LinearRegression import LinearRegression

def main():
	reg = LinearRegression()
	w, b = reg.fit('datas/data.csv', lr=0.01, iter=10000, visualize=False)
	
	mileage = 36000
	reg.predict(mileage, w, b)

if __name__ == "__main__":
	main()