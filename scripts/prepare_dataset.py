import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ===============================
# PATHS
# ===============================
RAW_DIR = "../data/raw/dataset"
IMAGE_DIR = os.path.join(RAW_DIR, "images")
LABELS_PATH = os.path.join(RAW_DIR, "labels.csv")
PROCESSED_DIR = "../data/processed"

# ===============================
# PARAMETERS
# ===============================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ===============================
# LOAD LABELS
# ===============================
df = pd.read_csv(LABELS_PATH)

label_map = {
    0: "honeybee",
    1: "bumblebee"
}

df["label"] = df["genus"].map(label_map)
df["filename"] = df["id"].astype(str) + ".jpg"

# ===============================
# CREATE DIRECTORIES
# ===============================
for split in ["train", "test"]:
    for cls in label_map.values():
        os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)

# ===============================
# SPLIT DATA
# ===============================
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["label"],
    random_state=RANDOM_STATE
)

# ===============================
# COPY IMAGES
# ===============================
def copy_images(dataframe, split):
    for _, row in dataframe.iterrows():
        src = os.path.join(IMAGE_DIR, row["filename"])
        dst = os.path.join(PROCESSED_DIR, split, row["label"], row["filename"])
        if os.path.exists(src):
            shutil.copy(src, dst)

copy_images(train_df, "train")
copy_images(test_df, "test")

print("Dataset successfully prepared.")
