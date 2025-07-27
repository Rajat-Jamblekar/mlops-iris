from sklearn.datasets import load_iris
import pandas as pd
def load_and_save_iris(filepath="data/iris.csv"):
    """
    Loads the Iris dataset from sklearn and saves it as a CSV file.
    
    Parameters:
    filepath (str): Path where the CSV file will be saved. Defaults to 'data/iris.csv'
    """
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(filepath, index=False)
    print(f"Iris dataset saved to {filepath}")
if __name__ == "__main__":
    load_and_save_iris()