import  argparse
import  pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt


def linearRegression(data):
    '''
    Perform linear regression on the provided data.
    Args:
        data (pd.DataFrame): DataFrame containing 'km' and 'price' columns.
    Returns:
        tuple: Contains theta0, theta1, min and max values of mileage and price.
    '''
    learningRate = 0.5
    iterations = 1000
    
    X_mileage = np.array(data['km'])
    X_mileage_n = (X_mileage - np.min(X_mileage)) / (np.max(X_mileage) - np.min(X_mileage))
    
    y_price = np.array(data['price'])
    y_price_n = (y_price - np.min(y_price)) / (np.max(y_price) - np.min(y_price))
    
    m = X_mileage_n.shape[0]
    if m == 0:
        raise ValueError("The data must contain at least one row.")
    
    theta0 = 0.0
    theta1 = 0.0
    
    print(X_mileage_n.shape)

    for i in range(iterations):
        Y = X_mileage_n * theta1 + theta0
        cost = (1 / (2 * m)) * np.sum((Y - y_price_n) ** 2)
        print(f"Iteration {i+1}/{iterations}, Cost: {cost}")
        tmp_theta0 = learningRate * ((1 / m) * np.sum(Y - y_price_n))
        theta0 -= tmp_theta0
        tmp_theta1 = learningRate * ((1 / m) * np.sum((Y - y_price_n) * X_mileage_n))
        theta1 -= tmp_theta1
        
    precision = calculate_precision(y_price_n, Y)
    print(f"Precision: {precision:.2f}%")
    
    return theta0, theta1, np.min(X_mileage), np.max(X_mileage), np.min(y_price), np.max(y_price)


def calculate_precision(y_actual, y_pred):
    '''
    Calculate the precision of the model.
    Args:
        y_actual (np.ndarray): Actual values.
        y_pred (np.ndarray): Predicted values.
    Returns:
        float: Precision of the model as a percentage.
    '''
    if len(y_actual) == 0 or len(y_pred) == 0:
        return 0.0
    up = np.sum((y_actual - y_pred) ** 2)
    down = np.sum((y_actual - np.mean(y_actual)) ** 2)
    precision = (1 - (up / down)) * 100
    if down == 0:
        return 0.0
    return precision


def main():
    '''
    Main function to handle command line arguments and train the linear regression model.
    It expects a CSV file with car mileage and prices.
    Args:
        data_file (str): Path to the CSV file containing car mileage and prices.
        values_file (str): Path to save the theta values.
    '''
    parser = argparse.ArgumentParser(description="Train a linear regression model to predict car prices based on mileage.")
    parser.add_argument('--data_file', '-d', type=str, required=True, help="Path to the data file containing car mileage and prices.")
    parser.add_argument('--values_file', '-f', type=str, default='values.csv', help="Path to save the values in a file.")
    args = parser.parse_args()

    try:
        data = pd.read_csv(args.data_file)
        if 'km' not in data.columns or 'price' not in data.columns:
            raise KeyError("The data file must contain 'km' and 'price' columns.")
        
        theta0, theta1, X_min, X_max, price_min, price_max = linearRegression(data)
        
        with open(args.values_file, mode='w') as file:
            file.write("theta0,theta1,min,max,price_min,price_max\n")
            file.write(f"{theta0},{theta1},{X_min},{X_max},{price_min},{price_max}\n")
        
        print(f"Model trained successfully. Values saved to {args.values_file}.")
        
        normalized_data = (data - data.min()) / (data.max() - data.min())
        plt.figure(figsize=(10, 6))
        plt.scatter(normalized_data['km'], normalized_data['price'], color='blue', label='Data Points')
        plt.plot(normalized_data['km'], normalized_data['km'] * theta1 + theta0, color='red', label='Regression Line')
        plt.xlabel('Mileage (km)')
        plt.ylabel('Price')
        plt.title('Car Price vs Mileage')
        plt.legend()
        plt.show()
        plt.savefig('regression_plot.png')
        print("Regression plot saved as 'regression_plot.png'.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()