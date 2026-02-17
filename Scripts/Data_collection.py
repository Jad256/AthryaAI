import pandas as pd
import logging
import os
import sys

sys.path.append(os.path.abspath('src'))

from config import DATA_PATH  # Assuming this is configured correctly in config.py
from utilities import setup_logging

    # Setup logging
setup_logging()

def load_and_explore_data(path):
        """ Load data and print basic statistics. """
        try:
            data = pd.read_csv(path)
            logging.info("Data loaded successfully.")
            
            logging.info(f"First 5 rows of the data:\n{data.head()}")
            logging.info(f"Data info:\n{data.info()}")
            logging.info(f"Data description:\n{data.describe()}")
            
            return data
        except Exception as e:
            logging.error(f"Failed to load data from {path}: {e}")
            raise

def clean_data(data):
        """ Clean data by dropping rows and columns with any missing values. """
        try:
            data_cleaned = data.dropna()
            logging.info("Rows with NaN values dropped.")
            
            # If you meant to drop columns with all NaN values, you need to specify 'axis=1' and 'how='all''
            # data_cleaned = data.dropna(axis=1, how='all')  
            # logging.info("Columns with all NaN values dropped.")

            logging.info(f"Data after cleaning:\n{data_cleaned.head()}")
            logging.info(f"Cleaned data info:\n{data_cleaned.info()}")
            logging.info(f"Cleaned data description:\n{data_cleaned.describe()}")

            return data_cleaned
        except Exception as e:
            logging.error("Failed to clean data: {e}")
            raise

if __name__ == "__main__":
        data_path = os.path.join(os.getcwd(), DATA_PATH)  # Ensure path is constructed correctly
        data = load_and_explore_data(data_path)
        clean_data(data)