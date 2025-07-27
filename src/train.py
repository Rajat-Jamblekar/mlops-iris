# # src/train.py

# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import mlflow
# import mlflow.sklearn
# import matplotlib.pyplot as plt

# def load_data():
#     X_train = pd.read_csv("data/X_train.csv")
#     X_test = pd.read_csv("data/X_test.csv")
#     y_train = pd.read_csv("data/y_train.csv").values.ravel()
#     y_test = pd.read_csv("data/y_test.csv").values.ravel()
#     return X_train, X_test, y_train, y_test

# def train_and_log_model(model, params, X_train, y_train, X_test, y_test, model_name):
#     with mlflow.start_run():
#         # Set tags and log parameters
#         mlflow.set_tag("model", model_name)
#         for param, value in params.items():
#             mlflow.log_param(param, value)

#         model.set_params(**params)
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)

#         # Log metrics and model
#         mlflow.log_metric("accuracy", acc)
#         mlflow.sklearn.log_model(model, "model")

#         plt.savefig("confusion_matrix.png")
#         mlflow.log_artifact("confusion_matrix.png")

#         print(f"{model_name} Accuracy: {acc:.4f}")
#         return acc, mlflow.active_run().info.run_id

# if __name__ == "__main__":
#     mlflow.set_experiment("iris_classification")
#     X_train, X_test, y_train, y_test = load_data()

#     # Model 1: Logistic Regression
#     lr_model = LogisticRegression()
#     lr_params = {"solver": "liblinear", "C": 1.0}
#     lr_acc, lr_run_id = train_and_log_model(lr_model, lr_params, X_train, y_train, X_test, y_test, "LogisticRegression")

#     # Model 2: Random Forest
#     rf_model = RandomForestClassifier()
#     rf_params = {"n_estimators": 100, "max_depth": 3}
#     rf_acc, rf_run_id = train_and_log_model(rf_model, rf_params, X_train, y_train, X_test, y_test, "RandomForest")

#     # Register best model
#     if rf_acc > lr_acc:
#         best_run_id = rf_run_id
#         model_name = "RandomForest"
#     else:
#         best_run_id = lr_run_id
#         model_name = "LogisticRegression"

#     model_uri = f"runs:/{best_run_id}/model"
#     mlflow.register_model(model_uri, "iris_model")
#     print(f"Best model '{model_name}' registered in MLflow.")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def train_and_log_model(model, params, X_train, y_train, X_test, y_test, model_name):
    with mlflow.start_run():
        # Set tags and log parameters
        mlflow.set_tag("model", model_name)
        for param, value in params.items():
            mlflow.log_param(param, value)

        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log metrics and model
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        print(f"{model_name} Accuracy: {acc:.4f}")
        return acc, mlflow.active_run().info.run_id

if __name__ == "__main__":
    mlflow.set_experiment("iris_classification")
    X_train, X_test, y_train, y_test = load_data()

    # Model 1: Logistic Regression
    lr_model = LogisticRegression()
    lr_params = {"solver": "liblinear", "C": 1.0}
    lr_acc, lr_run_id = train_and_log_model(lr_model, lr_params, X_train, y_train, X_test, y_test, "LogisticRegression")

    # Model 2: Random Forest
    rf_model = RandomForestClassifier()
    rf_params = {"n_estimators": 100, "max_depth": 3}
    rf_acc, rf_run_id = train_and_log_model(rf_model, rf_params, X_train, y_train, X_test, y_test, "RandomForest")

    # Register best model
    if rf_acc > lr_acc:
        best_run_id = rf_run_id
        model_name = "RandomForest"
    else:
        best_run_id = lr_run_id
        model_name = "LogisticRegression"

    model_uri = f"runs:/{best_run_id}/model"
    
    # Register the model without specifying a stage
    mlflow.register_model(model_uri, "iris_model")
    print(f"Best model '{model_name}' registered in MLflow.")

    # Optionally, you can log additional metadata or tags if needed
    # For example, you can log the model version or other relevant information
    # mlflow.set_tag("best_model", model_name)


