from sklearn.datasets import fetch_openml
from pathlib import Path
import sys

dataset_name = "covertype"
dataset_version = 3

output_filename = "dataset.csv"
file_path = Path(output_filename)

if file_path.is_file():
	print("You already have the dataset! Don't be greedy. Everybody gets ONE dataset.")
	sys.exit(0)

print("Downloading the dataset, please wait!")
dataset = fetch_openml(name=dataset_name, version=dataset_version, as_frame=True)

df = dataset.frame

df.to_csv(output_filename, index=False)

print("Successfully downloaded the dataset as dataset.csv")
