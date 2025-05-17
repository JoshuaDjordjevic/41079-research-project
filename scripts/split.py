import os
import shutil
import random
from pathlib import Path

def create_test_set(base_dir="data/tomato", test_ratio=0.1):
    """
    Since test dataset not included, create a test directory by taking
    portion from train and valid datasets
    
    Args:
        base_dir: Base directory containing train and valid folders
        test_ratio: Ratio of images to move to test set from each class
    """
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")
    test_dir = os.path.join(base_dir, "test")
    
    os.makedirs(test_dir, exist_ok=True)
    
    #disease classes
    disease_classes = [d for d in os.listdir(train_dir) 
                      if os.path.isdir(os.path.join(train_dir, d))]
    
    print(f"Found {len(disease_classes)} disease classes: {disease_classes}")
    
    for disease in disease_classes:
        #create test directory
        test_disease_dir = os.path.join(test_dir, disease)
        os.makedirs(test_disease_dir, exist_ok=True)
        
        train_disease_dir = os.path.join(train_dir, disease)
        train_images = [f for f in os.listdir(train_disease_dir) 
                       if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        num_train_to_move = int(len(train_images) * test_ratio)
        train_images_to_move = random.sample(train_images, num_train_to_move)
        print(f"Moving {num_train_to_move} images from train/{disease} to test/{disease}")
        
        #move images from train -> test
        for img in train_images_to_move:
            src = os.path.join(train_disease_dir, img)
            dst = os.path.join(test_disease_dir, img)
            shutil.move(src, dst)
        
        valid_disease_dir = os.path.join(valid_dir, disease)
        if os.path.exists(valid_disease_dir):
            valid_images = [f for f in os.listdir(valid_disease_dir) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            num_valid_to_move = int(len(valid_images) * test_ratio)
            valid_images_to_move = random.sample(valid_images, num_valid_to_move)
            
            print(f"Moving {num_valid_to_move} images from valid/{disease} to test/{disease}")
            #move the images
            for img in valid_images_to_move:
                src = os.path.join(valid_disease_dir, img)
                dst = os.path.join(test_disease_dir, img)
                shutil.move(src, dst)

if __name__ == "__main__":
    create_test_set()