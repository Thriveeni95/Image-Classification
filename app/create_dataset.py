# Creating 🤗 Dataset structure

# Import necessary libraries
from sklearn.model_selection import train_test_split  # For splitting dataset into train/val/test
from pathlib import Path  # For convenient path manipulations
import pandas as pd  # For reading and handling CSV data
import os  # For creating directories
from tqdm import tqdm  # For progress bars
import shutil  # For copying image files


# ─────────────────────────────────────────────────────────────────────────────
# 1. Split Data 
# ─────────────────────────────────────────────────────────────────────────────

# Read the CSV file containing image IDs and their corresponding breed labels.
# The CSV is expected to have at least two columns: 'id' (filename without .jpg) and 'breed'.
labels = pd.read_csv("../data/labels.csv")

# Separate out the image IDs (X) and labels (y)
X = labels.id        # e.g., ["b532e2e6f68876649639aa216ea4cddf", ...]
y = labels.breed     # e.g., ["affenpinscher", "afghan_hound", ...]

# First split: reserve 20% of the data for the test set. Use stratification to keep
# class proportions similar across splits, and set random_state for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=13,
    shuffle=True,
    stratify=y
)

# Second split: from the remaining 80% (train+val), reserve 10% for validation.
# That means validation set = 0.8 * 0.1 = 8% of the original, and train = 72% of original.
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,
    random_state=13,
    shuffle=True,
    stratify=y_train
)

# Combine the split arrays back into DataFrames for easier iteration later.
train_data = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
val_data   = pd.concat([X_val,   y_val],   axis=1).reset_index(drop=True)
test_data  = pd.concat([X_test,  y_test],  axis=1).reset_index(drop=True)

# Now:
#   train_data  is a DataFrame with columns ["id", "breed"] for the training examples
#   val_data    is a DataFrame with columns ["id", "breed"] for the validation examples
#   test_data   is a DataFrame with columns ["id", "breed"] for the test examples


# ─────────────────────────────────────────────────────────────────────────────
# 2. Make Dataset Dir and Splits
# ─────────────────────────────────────────────────────────────────────────────

# Define the root directory for the new ImageFolder‐style dataset.
# (It will be created relative to the current working directory.)
DatasetDir = Path("../PetClassification")

# Create the top‐level directory. If it already exists, this line will raise an error—
# you can wrap it in a try/except or use `exist_ok=True` if you want to run this script
# multiple times without crashing. For clarity, we’ll keep it as os.mkdir for now.
os.mkdir(DatasetDir)                  # Creates "../PetClassification"
os.mkdir(DatasetDir / "train")        # Creates "../PetClassification/train"
os.mkdir(DatasetDir / "validation")   # Creates "../PetClassification/validation"
os.mkdir(DatasetDir / "test")         # Creates "../PetClassification/test"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Make Classes (Breeds) Directories
# ─────────────────────────────────────────────────────────────────────────────

# For each unique breed label in the TRAIN split, create subdirectories under
# train/, validation/, and test/. We only use breeds from the training set to
# ensure every breed folder exists in all splits (even if validation/test has zero for some).
for breed in tqdm(train_data.breed.unique(), desc="Creating breed subfolders"):
    os.mkdir(DatasetDir / "train"      / breed)  # e.g. "../PetClassification/train/affenpinscher"
    os.mkdir(DatasetDir / "validation" / breed)  # e.g. "../PetClassification/validation/affenpinscher"
    os.mkdir(DatasetDir / "test"       / breed)  # e.g. "../PetClassification/test/affenpinscher"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Moving Images to Split Folder and Breed
# ─────────────────────────────────────────────────────────────────────────────

# Define the path where the original flat images are stored.
# Each image file is assumed to be named "<id>.jpg".
IMG_PATH = "../data/current_images"  # e.g. "../data/current_images/b532e2e6f68876649639aa216ea4cddf.jpg"

def make_split_folder(split_df: pd.DataFrame, split: str):
    """
    Copies each image from IMG_PATH into:
      ../PetClassification/<split>/<breed>/<id>.jpg

    Arguments:
      - split_df: DataFrame containing columns ["id", "breed"] for this split
      - split:    A string, one of "train", "validation", or "test", indicating which subfolder
                  to copy into.
    """
    # Iterate over every row in the DataFrame. tqdm provides a progress bar.
    for idx, row in tqdm(
        split_df.iterrows(),
        total=len(split_df),
        desc=f"Making {split} folder"
    ):
        img_name, breed = row["id"], row["breed"]

        # Build source file path. Example: "../data/current_images/b532e2e6f68876649639aa216ea4cddf.jpg"
        src_file = os.path.join(IMG_PATH, f"{img_name}.jpg")

        # Build destination file path. Example: "../PetClassification/train/affenpinscher/b532e2e6f68876649639aa216ea4cddf.jpg"
        dst_file = DatasetDir / split / breed / f"{img_name}.jpg"

        # Copy the image from source to destination.
        # If the source file does not exist, this will raise an error or you can choose to skip.
        shutil.copyfile(src_file, dst_file)


# Populate the "train" split:
make_split_folder(train_data, "train")

# Populate the "validation" split:
make_split_folder(val_data, "validation")

# Populate the "test" split:
make_split_folder(test_data, "test")
