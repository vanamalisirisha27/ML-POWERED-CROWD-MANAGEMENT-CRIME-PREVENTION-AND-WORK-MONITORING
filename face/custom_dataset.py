import os
import random
from PIL import Image
import numpy as np

def preprocess_custom_dataset(dataset_dir, target_dir, img_size=(224, 224), train_size=0.8, val_size=0.1, test_size=0.1):
    # Create target directories for training, validation, and testing sets
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Count the number of images for each person
    person_counts = {}
    for person in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person)
        if os.path.isdir(person_dir):
            person_counts[person] = len(os.listdir(person_dir))

    # Calculate the maximum count of images per person for oversampling
    max_count = max(person_counts.values())

    # Iterate through each person in the dataset directory
    for person, count in person_counts.items():
        person_dir = os.path.join(dataset_dir, person)
        if os.path.isdir(person_dir):
            # Get list of images for the current person
            images = os.listdir(person_dir)
            random.shuffle(images)

            # Calculate the number of images to oversample to match the maximum count
            oversample_count = max_count - count if count < max_count else 0

            # Sample images with replacement for oversampling
            oversampled_images = random.choices(images, k=oversample_count)

            # Combine original and oversampled images
            images.extend(oversampled_images)

            # Split the images into training, validation, and testing sets
            train_images, val_test_images = np.split(images, [int(train_size * len(images))])
            val_images, test_images = np.split(val_test_images, [int(val_size/(1-train_size) * len(val_test_images))])

            # Preprocess and save images to the corresponding directories
            for img_file in train_images:
                preprocess_and_save(os.path.join(person_dir, img_file), os.path.join(train_dir, person), img_size)
            for img_file in val_images:
                preprocess_and_save(os.path.join(person_dir, img_file), os.path.join(val_dir, person), img_size)
            for img_file in test_images:
                preprocess_and_save(os.path.join(person_dir, img_file), os.path.join(test_dir, person), img_size)

def preprocess_and_save(img_path, target_dir, img_size):
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Open and preprocess the image
    img = Image.open(img_path)
    img = img.resize(img_size)
    #img = np.array(img)

    # Save the preprocessed image
    img_filename = os.path.basename(img_path)
    img.save(os.path.join(target_dir, img_filename))

custom_dataset_dir = 'C:/crowdcrime/yolov8-streamlit-detection-tracking/face/custom_dataset'
preprocessed_target_dir = 'C:/crowdcrime/yolov8-streamlit-detection-tracking/face/preprocessed_custom_dataset'
preprocess_custom_dataset(custom_dataset_dir, preprocessed_target_dir)