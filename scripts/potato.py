import os
import shutil
from pathlib import Path

# Define source and target directories
source_root = Path("data/potato-leaf")
target_root = Path("data/potato")

# Create target root if it doesn't exist
target_root.mkdir(parents=True, exist_ok=True)

# Go through each dataset split
for split in ["train", "test", "valid"]:
    split_path = source_root / split
    if not split_path.exists():
        continue
    
    # Go through each class folder inside each split
    for class_folder in split_path.iterdir():
        if class_folder.is_dir():
            target_class_folder = target_root / class_folder.name
            target_class_folder.mkdir(parents=True, exist_ok=True)
            
            # Move all files to the target class folder
            for file in class_folder.iterdir():
                if file.is_file():
                    shutil.move(str(file), str(target_class_folder / file.name))