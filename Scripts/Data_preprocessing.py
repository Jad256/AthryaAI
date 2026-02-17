import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import logging
import os
import sys

# Add the src directory to Python's search path
sys.path.append(os.path.abspath('src'))

# Now do your imports
from utilities import setup_logging, load_data
from config import DATA_PATH



# Setup logging
setup_logging()

def preprocess_data():
    try:
        # Load data using utility function
        data = load_data(DATA_PATH)
        logging.info("Data loaded successfully.")

        # Handling Imbalance
        X = data.drop('Class', axis=1)
        y = data['Class']
        rus = RandomUnderSampler()
        X_res, y_res = rus.fit_resample(X, y)
        logging.info("Data resampling completed.")

        # Split into train-test sets and save processed data if needed
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
        logging.info("Train-test split completed.")

        # Feature Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("Feature scaling completed.")

        # Save only the resampled data
        save_data(X_res, 'X_res.pkl')
        save_data(y_res, 'Y_res.pkl')
        logging.info("Resampled data saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")
        raise

def save_data(data, filename):
    directory = 'data'
    if not os.path.exists(directory):
        os.makedirs(directory)  # Ensure directory exists
    path = os.path.join(directory, filename)  # Correct path assembly using os.path.join
    try:
        with open(path, 'wb') as file:
            pickle.dump(data, file)
        logging.info(f"{filename} saved at {path}")
    except IOError as e:
        logging.error(f"Failed to save {filename} due to an IOError: {e}")
        raise

if __name__ == "__main__":
    preprocess_data()
