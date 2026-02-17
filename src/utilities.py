import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    import pandas as pd
    return pd.read_csv(path)
