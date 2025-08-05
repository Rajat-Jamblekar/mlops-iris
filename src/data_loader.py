import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IrisDataLoader:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the Iris dataset"""
        try:
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['target'] = iris.target
            df['target_name'] = df['target'].map(dict(enumerate(iris.target_names)))
            
            # Save raw data
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/iris.csv', index=False)
            
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
            return df, iris.feature_names, iris.target_names
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, df, feature_names):
        """Preprocess the data"""
        try:
            # Features and target
            X = df[feature_names].values
            y = df['target'].values
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"Data preprocessed. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def get_preprocessed_data(self):
        """Get preprocessed data ready for training"""
        df, feature_names, target_names = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(df, feature_names)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'target_names': target_names,
            'scaler': self.scaler
        }

if __name__ == "__main__":
    loader = IrisDataLoader()
    data = loader.get_preprocessed_data()
    print("Data loading and preprocessing completed successfully!")
    print(f"Training set size: {data['X_train'].shape[0]}")
    print(f"Test set size: {data['X_test'].shape[0]}")