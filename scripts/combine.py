import os
import shutil
from sklearn.model_selection import train_test_split
import random
import glob

def combine_potato_datasets(potato_dir="data/potato", 
                           potato_leaf_dir="data/potato-leaf", 
                           output_dir="data/combined_potato"):
    """
    Combines the potato and potato-leaf datasets into a single unified dataset
    with train/valid/test splits.
    
    Args:
        potato_dir: Path to the potato dataset directory
        potato_leaf_dir: Path to the potato-leaf dataset directory
        output_dir: Path to the output directory for the combined dataset
    """
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Process potato dataset (which doesn't have train/valid/test splits)
    process_potato_dataset(potato_dir, output_dir)
    
    # Process potato-leaf dataset (which already has train/valid/test splits)
    process_potato_leaf_dataset(potato_leaf_dir, output_dir)
    
    print(f"Datasets combined successfully at {output_dir}")
    
    # Count and display classes and images in the combined dataset
    count_and_display_stats(output_dir)

def process_potato_dataset(potato_dir, output_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    """
    Process the potato dataset and distribute images to train/valid/test splits
    """
    classes = [d for d in os.listdir(potato_dir) if os.path.isdir(os.path.join(potato_dir, d))]
    
    for class_name in classes:
        print(f"Processing potato class: {class_name}")
        
        # Create class directories in each split
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(output_dir, split, f"potato_{class_name}"), exist_ok=True)
        
        # Get all images in this class
        img_paths = glob.glob(os.path.join(potato_dir, class_name, "*.jpg")) + \
                   glob.glob(os.path.join(potato_dir, class_name, "*.jpeg")) + \
                   glob.glob(os.path.join(potato_dir, class_name, "*.png"))
        
        # Split into train, valid, test
        train_imgs, test_valid_imgs = train_test_split(img_paths, test_size=(valid_ratio + test_ratio), random_state=42)
        valid_imgs, test_imgs = train_test_split(test_valid_imgs, test_size=test_ratio/(valid_ratio + test_ratio), random_state=42)
        
        # Copy images to respective directories
        for img_path, split_dir in zip([train_imgs, valid_imgs, test_imgs], ['train', 'valid', 'test']):
            for img in img_path:
                shutil.copy2(img, os.path.join(output_dir, split_dir, f"potato_{class_name}", os.path.basename(img)))

def process_potato_leaf_dataset(potato_leaf_dir, output_dir):
    """
    Process the potato-leaf dataset which already has train/valid/test splits
    """
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        if not os.path.exists(os.path.join(potato_leaf_dir, split)):
            if split == 'test' and os.path.exists(os.path.join(potato_leaf_dir, 'val')):
                # Handle case where validation is named 'val' instead of 'valid'
                actual_split = 'val'
            else:
                print(f"Warning: {split} directory not found in {potato_leaf_dir}")
                continue
        else:
            actual_split = split
            
        split_dir = os.path.join(potato_leaf_dir, actual_split)
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        for class_name in classes:
            print(f"Processing potato-leaf {split} class: {class_name}")
            
            # Create class directory in output
            os.makedirs(os.path.join(output_dir, split, f"leaf_{class_name}"), exist_ok=True)
            
            # Get all images in this class
            img_paths = glob.glob(os.path.join(split_dir, class_name, "*.jpg")) + \
                       glob.glob(os.path.join(split_dir, class_name, "*.jpeg")) + \
                       glob.glob(os.path.join(split_dir, class_name, "*.png"))
            
            # Copy images to respective directories
            for img in img_paths:
                shutil.copy2(img, os.path.join(output_dir, split, f"leaf_{class_name}", os.path.basename(img)))

def count_and_display_stats(combined_dir):
    """
    Count and display statistics about the combined dataset
    """
    splits = ['train', 'valid', 'test']
    total_images = 0
    
    print("\nCombined Dataset Statistics:")
    print("----------------------------")
    
    for split in splits:
        split_dir = os.path.join(combined_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        split_total = 0
        
        print(f"\n{split.capitalize()} split:")
        for class_name in classes:
            img_count = len(glob.glob(os.path.join(split_dir, class_name, "*.jpg"))) + \
                        len(glob.glob(os.path.join(split_dir, class_name, "*.jpeg"))) + \
                        len(glob.glob(os.path.join(split_dir, class_name, "*.png")))
            print(f"  - {class_name}: {img_count} images")
            split_total += img_count
            
        print(f"  Total {split} images: {split_total}")
        total_images += split_total
    
    print(f"\nTotal dataset images: {total_images}")

# Run the combination process if executed directly
if __name__ == "__main__":
    combine_potato_datasets()