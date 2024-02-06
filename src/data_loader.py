import requests
import os
import pandas as pd

def download_file(url, destination_path):
    """Download a file from a specified URL to a destination directory."""
    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()
    with open(destination_path, 'wb') as f:
        f.write(response.content)

def load_data(file_path, delimiter=',', encoding='utf-8'):
    """Load and return a DataFrame from a CSV file using the specified delimiter."""
    return pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, engine='python', quotechar='"')

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(base_dir, exist_ok=True)

    # URLs for the train and test datasets
    train_url = 'https://drive.google.com/uc?export=download&id=1Q_FF3ozMj4ReG7Ejy_eArFGAetd4pju0'
    test_url = 'https://drive.google.com/uc?export=download&id=1WGv5UWu8ayziZSbrxZxa5GTLMet8erwy'

    # File paths for the downloaded CSV files
    train_file_path = os.path.join(base_dir, 'train.csv')
    test_file_path = os.path.join(base_dir, 'test.csv')

    # Download the files
    download_file(train_url, train_file_path)
    download_file(test_url, test_file_path)

    # Load the data into pandas DataFrames
    train_df = load_data(train_file_path)
    test_df = load_data(test_file_path)
