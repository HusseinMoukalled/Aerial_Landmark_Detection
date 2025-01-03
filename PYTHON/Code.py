import os
import cv2


class_mapping = {'plane': 0, 'storage-tank': 1, 'small-vehicle': 2, 'large-vehicle': 3, 'ship': 4, 'harbor': 5, 'ground-track-field': 6, 'soccer-ball-field': 7, 'tennis-court': 8, 'swimming-pool': 9, 'baseball-diamond': 10, 'roundabout': 11, 'basketball-court': 12, 'bridge': 13, 'helicopter': 14, 'container-crane': 15}


def convert_to_yolo_format(annotation_path, image_path):
    # Get image dimensions
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    with open(annotation_path, 'r') as f:
        lines = f.readlines()[2:]  # Skip metadata lines

    yolo_annotations = []

    for line in lines:
        coords = line.strip().split()


        try:

            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, coords[:8])  # First 8 are coordinates
            classname = coords[8]  # The 9th item is the class name
        except ValueError:
            print(f"Skipping invalid annotation in {annotation_path}: {line}")
            continue

        # Convert class name to class ID using the mapping
        class_id = class_mapping.get(classname, -1)  # Use -1 for unknown class names

        if class_id == -1:
            print(f"Warning: Unknown class '{classname}' found. Skipping annotation.")
            continue

        # Calculate bounding box (x_min, y_min, x_max, y_max)
        xmin = min(x1, x2, x3, x4)
        xmax = max(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        ymax = max(y1, y2, y3, y4)

        # Calculate YOLO format: center_x, center_y, width, height
        center_x = (xmin + xmax) / 2 / image_width
        center_y = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        # Ensure the coordinates are within bounds (0 to 1)
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        yolo_annotations.append(f"{class_id} {center_x} {center_y} {width} {height}")

    return yolo_annotations


image_folder = r"D:\ObjectDetection\images"  # Path to your image folder
annotation_folder = r"D:\ObjectDetection\DOTA-v1.5_train"  # Path to your annotation folder
output_annotation_folder = r"D:\ObjectDetection\yolo_annotations"  # New folder for YOLO annotations


if not os.path.exists(output_annotation_folder):
    os.makedirs(output_annotation_folder)

# Loop through images and convert annotations
for image_name in os.listdir(image_folder):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Check image extensions
        image_path = os.path.join(image_folder, image_name)
        annotation_path = os.path.join(annotation_folder, image_name.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt'))

        if os.path.exists(annotation_path):
            yolo_annotations = convert_to_yolo_format(annotation_path, image_path)

            # Save YOLO formatted annotations to the new folder
            yolo_annotation_path = os.path.join(output_annotation_folder, image_name.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt'))
            with open(yolo_annotation_path, 'w') as f:
                for ann in yolo_annotations:
                    f.write(ann + '\n')

import os
import shutil

def store_and_remove_large_images_and_annotations(images_folder, annotations_folder, large_images_folder, large_annotations_folder, max_image_size_mb):

    # Convert max size to bytes
    max_size_bytes = max_image_size_mb * 1024 * 1024

    # List all files in the images folder
    images = os.listdir(images_folder)
    stored_files = []

    for image_file in images:
        image_path = os.path.join(images_folder, image_file)


        if not os.path.isfile(image_path):
            continue

        # Get the file size
        image_size = os.path.getsize(image_path)

        if image_size > max_size_bytes:
            # Construct the corresponding annotation file path
            annotation_file = os.path.splitext(image_file)[0] + ".txt"
            annotation_path = os.path.join(annotations_folder, annotation_file)

            # Create the new paths for large images and annotations
            large_image_path = os.path.join(large_images_folder, image_file)
            large_annotation_path = os.path.join(large_annotations_folder, annotation_file)

            # Copy the image to the new folder
            shutil.copy(image_path, large_image_path)
            stored_files.append(large_image_path)

            # Copy the annotation file to the new folder if it exists
            if os.path.exists(annotation_path):
                shutil.copy(annotation_path, large_annotation_path)
                stored_files.append(large_annotation_path)

            # Remove the image from the original folder after copying it
            os.remove(image_path)

            # Remove the corresponding annotation if it exists and was copied
            if os.path.exists(annotation_path):
                os.remove(annotation_path)

    print(f"Stored and removed {len(stored_files)} large files:")
    for file in stored_files:
        print(file)



dataset_path = '/content/drive/MyDrive/yolov5/dataset'
images_folder = os.path.join(dataset_path, 'images/train-images')
annotations_folder = os.path.join(dataset_path, 'labels/train-annotations')

# Folder to store large images and their annotations
large_images_folder = os.path.join(dataset_path, 'large_images')
large_annotations_folder = os.path.join(dataset_path, 'large_annotations')

# Make sure the new folders exist
os.makedirs(large_images_folder, exist_ok=True)
os.makedirs(large_annotations_folder, exist_ok=True)


max_image_size_mb = 10

# Call the function
store_and_remove_large_images_and_annotations(images_folder, annotations_folder, large_images_folder, large_annotations_folder, max_image_size_mb)

import os

# Path to dataset in Google Drive
dataset_path = '/content/drive/MyDrive/yolov5/dataset'
images_path = os.path.join(dataset_path, 'images/train-images')
labels_path = os.path.join(dataset_path, 'labels/train-annotations')

import os
import shutil
import random

# Paths for dataset
dataset_path = '/content/drive/MyDrive/yolov5/dataset'
images_path = os.path.join(dataset_path, 'images/train-images')
labels_path = os.path.join(dataset_path, 'labels/train-annotations')

# Create directories for splits
train_images_path = os.path.join(dataset_path, 'images/train.split')
val_images_path = os.path.join(dataset_path, 'images/val.split')
test_images_path = os.path.join(dataset_path, 'images/test.split')

train_labels_path = os.path.join(dataset_path, 'labels/train.split')
val_labels_path = os.path.join(dataset_path, 'labels/val.split')
test_labels_path = os.path.join(dataset_path, 'labels/test.split')

os.makedirs(train_images_path, exist_ok=True)
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)

os.makedirs(train_labels_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)
os.makedirs(test_labels_path, exist_ok=True)

# Get all image filenames
all_images = [img for img in os.listdir(images_path) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle and split the dataset
random.seed(42)
random.shuffle(all_images)

# Split dataset into 70% train, 20% val, 10% test
train_split = int(len(all_images) * 0.7)
val_split = int(len(all_images) * 0.9)  # 70% + 20%

train_images = all_images[:train_split]
val_images = all_images[train_split:val_split]
test_images = all_images[val_split:]

# Function to move files
def move_files(images, src_img_path, src_lbl_path, dest_img_path, dest_lbl_path):
    for img in images:
        # Move images
        shutil.copy2(os.path.join(src_img_path, img), os.path.join(dest_img_path, img))
        # Move corresponding labels
        label_file = img.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        shutil.copy2(os.path.join(src_lbl_path, label_file), os.path.join(dest_lbl_path, label_file))

# Move the files into respective directories
move_files(train_images, images_path, labels_path, train_images_path, train_labels_path)
move_files(val_images, images_path, labels_path, val_images_path, val_labels_path)
move_files(test_images, images_path, labels_path, test_images_path, test_labels_path)

# Print summary
print(f"Training set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")
print(f"Testing set: {len(test_images)} images")


!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
!pip install -r requirements.txt  # install


from ultralytics import YOLO

model = YOLO('yolov5m.pt')

results = model.train(data='/content/drive/MyDrive/yolov5/dataset/data.yaml', epochs=15, imgsz=960, batch=4, freeze=10, augment=True)


import shutil

# Source and destination paths
source = "runs/detect/train3"
destination = "/content/drive/MyDrive/yolov5/dataset/train.results"

# Copy the folder
shutil.copytree(source, destination, dirs_exist_ok=True)
print("Files copied successfully!")


# Load the previously trained model
model = YOLO('runs/detect/train3/weights/last.pt')

model.train(
    data='/content/drive/MyDrive/yolov5/dataset/data.yaml',
    epochs=10,
    imgsz=960,
    batch=4,
    augment=True,

)


import shutil

# Source and destination paths
source = "runs/detect/train4"
destination = "/content/drive/MyDrive/yolov5/dataset/train.results2"

# Copy the folder
shutil.copytree(source, destination, dirs_exist_ok=True)
print("Files copied successfully!")


import os
from pathlib import Path
from ultralytics import YOLO


model_best = YOLO('runs/detect/train4/weights/best.pt')

# Path to the images for inference
image_paths = [str(path) for path in Path('/content/drive/MyDrive/yolov5/dataset/images/test.split').glob('*.png')]

# Run inference and save annotated images and predictions
for image_path in image_paths:
    results_test = model_best.predict(
        source=image_path,
        save=True,
        save_txt=True,
        imgsz=960,
        conf=0.5
    )



import shutil

# Source and destination paths
source = "runs/detect/predict"
destination = "/content/drive/MyDrive/yolov5/dataset/predict.results"

# Copy the folder
shutil.copytree(source, destination, dirs_exist_ok=True)
print("Files copied successfully!")


import os


def iou(box1, box2):
    # Convert from (center_x, center_y, width, height) to (xmin, ymin, xmax, ymax)
    box1 = [box1[0] - box1[2]/2, box1[1] - box1[3]/2, box1[0] + box1[2]/2, box1[1] + box1[3]/2]
    box2 = [box2[0] - box2[2]/2, box2[1] - box2[3]/2, box2[0] + box2[2]/2, box2[1] + box2[3]/2]

    # Calculate the area of intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate the area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the area of union
    union_area = box1_area + box2_area - intersection_area

    # Return IoU
    return intersection_area / union_area if union_area > 0 else 0


# Function to compare predicted annotations with ground truth
def compare_annotations(predicted_folder, ground_truth_folder, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0

    # Loop through each predicted annotation file
    for pred_file in os.listdir(predicted_folder):
        if pred_file.endswith('.txt'):
            gt_file = os.path.join(ground_truth_folder, pred_file)

            # If no ground truth file exists for the predicted file, skip it
            if not os.path.exists(gt_file):
                print(f"Ground truth not found for {pred_file}")
                continue

            # Read predicted bounding boxes (each row: class_id, center_x, center_y, width, height)
            with open(os.path.join(predicted_folder, pred_file), 'r') as f:
                pred_boxes = [list(map(float, line.strip().split()[1:])) for line in f.readlines()]

            # Read ground truth bounding boxes (same format as above)
            with open(gt_file, 'r') as f:
                gt_boxes = [list(map(float, line.strip().split()[1:])) for line in f.readlines()]

            # Now, check for matching ground truth boxes for each predicted box based on IoU
            matched_gt = []  # List of ground truth boxes that have been matched
            for gt in gt_boxes:
                matched = False
                for pred in pred_boxes:
                    # Check IoU between predicted box and ground truth box
                    if iou(gt, pred) >= iou_threshold:
                        tp += 1
                        matched = True
                        matched_gt.append(gt)
                        break
                if not matched:
                    fn += 1  # No match for this ground truth box

            # Any remaining predicted boxes that don't match a ground truth are false positives
            fp += len(pred_boxes) - len(matched_gt)  # Unmatched predicted boxes

    # Calculate Precision and Recall
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    return tp, fp, fn, precision, recall


# Example usage:
predicted_folder = "/content/drive/MyDrive/yolov5/dataset/predict.results/labels"
ground_truth_folder = "/content/drive/MyDrive/yolov5/dataset/labels/test.split"

tp, fp, fn, precision, recall = compare_annotations(predicted_folder, ground_truth_folder)

