import os
import urllib.request
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
OUTPUT_FILE = "Mall_Customers.csv"

def fetch_data():
    print(f"Downloading dataset from {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, OUTPUT_FILE)
    df = pd.read_csv(OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return df

if __name__ == "__main__":
    fetch_data()
