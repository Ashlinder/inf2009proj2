import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# Paths and Configuration
DATA_PATH = "DCSASS_Dataset"
LABELS_PATH = os.path.join(DATA_PATH, "Labels")
ACTIVITIES = [folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder)) and folder != 'Labels']

# Collect all data and labels
all_data = []
all_labels = set()

for activity in ACTIVITIES:
    activity_folder = os.path.join(DATA_PATH, activity)
    label_file = os.path.join(LABELS_PATH, f"{activity}.csv")

    if not os.path.exists(label_file):
        print(f"Label file not found: {label_file}")
        continue

    df = pd.read_csv(label_file, sep=',', header=None)

    if df.shape[1] == 3:
        df.columns = ['filename', 'activity', 'label']
        all_labels.update(df['filename'].tolist())
    else:
        print(f"Unexpected number of columns in {label_file}")
        continue

    for _, row in df.iterrows():
        intermediate_folder = os.path.join(activity_folder, row['filename'].rsplit('_', 1)[0] + ".mp4")
        video_file = os.path.join(intermediate_folder, row['filename'] + ".mp4")

        if os.path.exists(video_file) and row['filename'] in all_labels:
            all_data.append((video_file, row['label']))
        else:
            print(f"File not found or label missing: {video_file}")

print(f"Total files collected: {len(all_data)}")

# Filter out entries with NaN labels
all_data = [(video, label) for video, label in all_data if pd.notna(label)]

# Check if data is empty
if len(all_data) == 0:
    raise ValueError("No data collected! Please check data paths and labels.")

# Ensure stratify list matches the length of all_data
stratify_labels = [d[1] for d in all_data]
if len(stratify_labels) != len(all_data):
    raise ValueError("Mismatch between data and labels. Please check your dataset.")

# Split the data
train_data, temp_data = train_test_split(all_data, test_size=0.3, random_state=42, stratify=stratify_labels)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42, stratify=[d[1] for d in temp_data])

# Save files
def save_data(data, folder):
    os.makedirs(folder, exist_ok=True)
    for video_file, label in data:
        dest_file = os.path.join(folder, os.path.basename(video_file))
        if not os.path.exists(dest_file):
            shutil.copy(video_file, dest_file)

# Save datasets
save_data(train_data, "data/train_set")
save_data(val_data, "data/val_set")
save_data(test_data, "data/test_set")

print(f"Data successfully split and saved: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")