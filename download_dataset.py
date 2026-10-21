from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import pandas as pd
from utils import downsample

dataset_name = "covertype"
dataset_version = 3

output_filename = "data/dataset.csv"
subset_filename = "data/sample.csv"
split_path = "data/"

random_state = 42

file_path = Path(output_filename)
subset_path = Path(subset_filename)

df = None

# Download or load the original (source) dataset...

if file_path.is_file():
	print("You already have the dataset! Don't be greedy. Everybody gets ONE dataset.")
	df = pd.read_csv(file_path)
else:
	print("Downloading the dataset, please wait!")
	dataset = fetch_openml(name=dataset_name, version=dataset_version, as_frame=True)

	df = dataset.frame

	df.to_csv(output_filename, index=False)

	print(f"Successfully downloaded the dataset, and saved it as {output_filename}")

print(f"Source dataset size: {len(df)}")

frequencies = df["class"].value_counts().sort_index()
print(f"Relative class support: {frequencies}")

# Make the reduced dataset...

print(f"Generating the downsamples dataset (cutoff at n=2000, random downsample) as {subset_filename}...")

# a general rule of thumb is that as long as
# the largest class is < 2x the smallest, we're good.
# smallest is ~2700, so 5400 is OK.

df2 = downsample(df,column_name="class",N=5400,random_state=42)

print(f"Successfully saved the data subset as {subset_filename}, with a total of {len(df2)} entries.")

df2.to_csv(subset_path, index=False)

# Train test split (to make it deterministic, just do it once!)

X = df2.drop(columns="class")
y = df2["class"]

# 100 --> 80 train, 20 test

print("Performing train/test/validation split...")

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size = 0.2,
	random_state = random_state
)

# 80 train --> 60 train, 20 validation

X_train, X_valid, y_train, y_valid = train_test_split(
	X_train,
	y_train,
	test_size = 0.25,
	random_state = random_state
)

print(f"Success! {len(X_train)} train, {len(X_test)} test, {len(X_valid)} validation.")

# Now we have 60 train, 20 valid, 20 test :)

# Stitch 'em back together for safe keeping...
X_train["class"] = y_train
X_valid["class"] = y_valid
X_test["class"] = y_test

X_train.to_csv(Path(split_path+"train.csv"), index=False)
X_valid.to_csv(Path(split_path+"valid.csv"), index=False)
X_test.to_csv(Path(split_path+"test.csv"), index=False)

print(f"Finished saving the training, validation, and test subsets in the {split_path} directory!")