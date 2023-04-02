from LinearRegression import LinearRegression
from utils import read_csv_file, show_graph

def main():
	dataset = read_csv_file('data.csv')
	# show_graph(dataset)
	reg = LinearRegression(lr=0.001, n_iters=1500)
	theta0, theta1 = reg.train(dataset)
	
	mileage = 36000
	price = reg.predict(theta0, theta1, mileage)
	print("\ntheta0 =", theta0)
	print("theta1 =", theta1)
	print(f"Predicted price for mileage {mileage} : {price}")
	dataset.append([mileage, price])

	show_graph(dataset)

if __name__ == "__main__":
    main()