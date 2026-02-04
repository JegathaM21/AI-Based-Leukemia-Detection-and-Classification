import os
import cv2
import numpy as np
from tqdm import tqdm
import random

# Set paths
DATASET_PATH = r"C:\Users\91787\Desktop\ProjectLeukemia\DataSet\train"
OUTPUT_PATH = r"C:\Users\91787\Desktop\ProjectLeukemia\DataSet\processed_train"

# Define output image size
IMAGE_SIZE = (224, 224)

# Number of augmented images to generate per original image
AUGMENT_COUNT = 3

def apply_augmentations(image):
    aug_images = []

    for _ in range(AUGMENT_COUNT):
        img_aug = image.copy()

        # Random rotation
        angle = random.randint(-20, 20)
        M = cv2.getRotationMatrix2D((IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2), angle, 1)
        img_aug = cv2.warpAffine(img_aug, M, IMAGE_SIZE)

        # Random horizontal flip
        if random.random() > 0.5:
            img_aug = cv2.flip(img_aug, 1)

        # Random brightness adjustment
        value = random.randint(-30, 30)
        img_aug = cv2.convertScaleAbs(img_aug, alpha=1, beta=value)

        # Random Gaussian noise
        noise = np.random.normal(0, 5, img_aug.shape).astype(np.uint8)
        img_aug = cv2.add(img_aug, noise)

        aug_images.append(img_aug)

    return aug_images

# Create output folder if not exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Loop through folders
for leukemia_type in os.listdir(DATASET_PATH):
    type_folder = os.path.join(DATASET_PATH, leukemia_type)

    if not os.path.isdir(type_folder):
        continue

    output_type_folder = os.path.join(OUTPUT_PATH, leukemia_type)
    os.makedirs(output_type_folder, exist_ok=True)

    for img_name in tqdm(os.listdir(type_folder), desc=f"Processing {leukemia_type}"):
        img_path = os.path.join(type_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping: {img_name}")
            continue

        img_resized = cv2.resize(img, IMAGE_SIZE)
        img_filtered = cv2.GaussianBlur(img_resized, (5, 5), 0)

        # Save original preprocessed image
        output_img_path = os.path.join(output_type_folder, img_name)
        cv2.imwrite(output_img_path, img_filtered)

        # Generate and save augmentations
        aug_images = apply_augmentations(img_filtered)
        for i, aug in enumerate(aug_images):
            aug_filename = f"{os.path.splitext(img_name)[0]}_aug{i+1}.jpg"
            aug_path = os.path.join(output_type_folder, aug_filename)
            cv2.imwrite(aug_path, aug)
