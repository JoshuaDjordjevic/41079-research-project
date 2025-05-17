import os
import shutil
import random

def create_train_valid_test_split(
    data_dir="data/potato",
    train_ratio=0.7,
    valid_ratio=0.2,
    test_ratio=0.1
):
    # Ensure ratios sum to 1
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6
    
    # Get all class subfolders
    classes = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    
    # Create output directories
    for split in ["train", "valid", "test"]:
        for c in classes:
            os.makedirs(os.path.join(data_dir, split, c), exist_ok=True)
    
    # For each class subfolder, partition images
    for c in classes:
        class_path = os.path.join(data_dir, c)
        all_images = [
            f for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ]
        
        random.shuffle(all_images)
        total = len(all_images)
        train_cutoff = int(train_ratio * total)
        valid_cutoff = int((train_ratio + valid_ratio) * total)
        
        train_images = all_images[:train_cutoff]
        valid_images = all_images[train_cutoff:valid_cutoff]
        test_images = all_images[valid_cutoff:]
        
        # Move files into respective directories
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(data_dir, "train", c, img)
            shutil.move(src, dst)
        
        for img in valid_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(data_dir, "valid", c, img)
            shutil.move(src, dst)
        
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(data_dir, "test", c, img)
            shutil.move(src, dst)

if __name__ == "__main__":
    create_train_valid_test_split()