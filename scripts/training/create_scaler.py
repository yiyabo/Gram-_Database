import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

X_train_path = "/Users/apple/AIBD/Gram-_Database/data/X_train.npy"
scaler_output_path = "/Users/apple/AIBD/Gram-_Database/model/scaler.pkl"

try:
    # Load training data
    X_train = np.load(X_train_path)
    print(f"Successfully loaded X_train.npy with shape: {X_train.shape}")

    if X_train.ndim != 2 or X_train.shape[1] == 0:
        print(f"Error: X_train.npy does not have the expected 2D shape or has zero features. Shape: {X_train.shape}")
    else:
        # Create and fit the scaler
        scaler = StandardScaler()
        print("Fitting StandardScaler on X_train data...")
        scaler.fit(X_train)
        print("StandardScaler fitted successfully.")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(scaler_output_path), exist_ok=True)

        # Save the scaler
        with open(scaler_output_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"StandardScaler saved to {scaler_output_path}")

except FileNotFoundError:
    print(f"Error: Training data file {X_train_path} not found.")
except Exception as e:
    print(f"An error occurred: {e}")
