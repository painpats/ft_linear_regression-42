import argparse
import csv


def predict_price(theta0, theta1, mileage):
    '''
    Predict the price of a car based on its mileage using the linear regression model.
    Args:
        theta0 (float): Intercept of the linear regression model.
        theta1 (float): Slope of the linear regression model.
        mileage (float): Mileage of the car.
    Returns:
        float: Estimated price of the car.
    '''
    estimated_price = theta0 + (theta1 * mileage)
    return estimated_price


def load_theta(values_file):
    '''
    Load theta values from a CSV file.
    Args:
        values_file (str): Path to the CSV file containing theta values.
    Returns:
        tuple: Contains theta0, theta1, min and max values of mileage.
    '''
    try:
        with open(values_file, mode='r') as file:
            reader = csv.DictReader(file)
            row = next(reader)
            theta0 = float(row['theta0'])
            theta1 = float(row['theta1'])
            min = float(row['min'])
            max = float(row['max'])
            price_min = float(row['price_min'])
            price_max = float(row['price_max'])
            return theta0, theta1, min, max, price_min, price_max
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {values_file} does not exist.")
    except ValueError:
        raise ValueError("The file does not contain valid numeric values for theta0 and theta1.")
    except KeyError:
        raise KeyError("The file does not contain the required columns theta0 and theta1.")
    

def main():
    '''
    Main function to handle command line arguments and predict the price of a car based on its mileage.
    It expects a CSV file with theta values.
    Args:
        values_file (str): Path to the CSV file containing theta values.
    '''
    parser = argparse.ArgumentParser(description="Predict the price of a car based on its mileage.")
    parser.add_argument('--values_file', '-f', type=str, nargs='?', default=None, help="Path to the values file.")
    args = parser.parse_args()
    theta0, theta1, min, max = 0, 0, 0, 0
    if args.values_file:
        try:
            theta0, theta1, min, max, price_min, price_max = load_theta(args.values_file)
            print(f"Loaded theta values: theta0 = {theta0}, theta1 = {theta1}, min = {min}, max = {max}")
        except Exception as e:
            print(f"Error loading values file: {e}")
            print("Using default values: theta0 = 0, theta1 = 0, min = 0, max = 1")
    try:
        mileage = float(input("Enter the mileage of the car: "))
        if max != min:
            mileage_n = (mileage - min) / (max - min)
        else:
            mileage_n = mileage
        estimated_price = predict_price(theta0, theta1, mileage_n)
        real_price = estimated_price * (price_max - price_min) + price_min
        print(f"The estimated price of the car is: {real_price}")
    except ValueError:
        print("Invalid input. Please enter a numeric value for mileage.")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")


if __name__ == "__main__":
    main()
