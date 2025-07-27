import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(save_path="data/"):
    # Load
    iris = load_iris(as_frame=True)
    df = iris.frame
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optional: Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # Save
    X_train_scaled.to_csv(f"{save_path}/X_train.csv", index=False)
    X_test_scaled.to_csv(f"{save_path}/X_test.csv", index=False)
    y_train.to_csv(f"{save_path}/y_train.csv", index=False)
    y_test.to_csv(f"{save_path}/y_test.csv", index=False)

    print(f"Preprocessed data saved to {save_path}")

if __name__ == "__main__":
    load_and_preprocess()
