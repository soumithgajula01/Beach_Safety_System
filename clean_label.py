import os
from glob import glob

DATASET_PATH = "/home/user/Desktop/Soumith/project"

def clean_label_file(label_path):
    changed = False
    cleaned_lines = []
    
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        # Skip empty lines
        if len(parts) < 5:
            continue

        cls = int(parts[0])
        if cls not in [0, 1]:   # Only 2 classes allowed
            changed = True
            continue

        # Ensure values are floats between 0–1
        try:
            vals = list(map(float, parts[1:5]))
        except:
            changed = True
            continue

        if any(v < 0 or v > 1 for v in vals):
            changed = True
            continue

        cleaned_lines.append(line)

    if changed:
        with open(label_path, "w") as f:
            f.writelines(cleaned_lines)

    return changed


folders = ["train/labels", "valid/labels", "test/labels"]

total_fixed = 0

for folder in folders:
    folder_path = os.path.join(DATASET_PATH, folder)
    for label_file in glob(folder_path + "/*.txt"):
        if clean_label_file(label_file):
            print("Fixed:", label_file)
            total_fixed += 1

print(f"\n✅ Cleaning completed. {total_fixed} label files fixed.")
