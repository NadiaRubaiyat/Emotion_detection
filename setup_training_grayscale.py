import torch
from torchvision import transforms
import os
from tabulate import tabulate
import pandas as pd
import numpy as np
from PIL import Image

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def summarize_results(test_accuracies, test_reports, training_times, dataset, path):
    summary_rows = []
    for model_name, test_accuracy in test_accuracies.items():
        report = test_reports[model_name]
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']
        train = training_times[model_name]
        summary_rows.append({
            "Model": model_name,
            "Accuracy": test_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
            "Training Time": train
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{path}/model_results.csv", index=False)
    print(f"\n{dataset} {path.capitalize()} Results Summary:")
    print(tabulate(summary_rows, headers="keys", tablefmt="pretty"))

class TransformWithLandmarks:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, landmarks):
        for t in self.transform.transforms:
            if isinstance(t, transforms.Resize):
                image, landmarks = self.resize(image, landmarks, t.size)
            elif isinstance(t, transforms.RandomHorizontalFlip):
                image, landmarks = self.horizontal_flip(image, landmarks, t.p)
            elif isinstance(t, transforms.RandomRotation):
                image, landmarks = self.rotate(image, landmarks, t.degrees)
            elif isinstance(t, transforms.RandomResizedCrop):
                image, landmarks = self.random_resized_crop(image, landmarks, t.size, t.scale)
            elif isinstance(t, transforms.ToTensor):
                image = t(image)  # Landmarks unaffected
            elif isinstance(t, transforms.Normalize):
                image = t(image)  # Landmarks unaffected
        
        landmarks = self.normalize_landmarks(landmarks, image)
        return image, landmarks
    
    def normalize_landmarks(self, landmarks, image):
        """Normalize landmarks to be within [0, 1]."""
        _, height, width = image.size() if image.ndimension() == 3 else image.shape
        # Normalize landmarks
        landmarks[:, 0] /= width
        landmarks[:, 1] /= height
        return landmarks

    def resize(self, image, landmarks, size):
        # Resize the image
        
        image = image.resize(size, Image.BILINEAR)
        return image, landmarks

    def horizontal_flip(self, image, landmarks, p=0.5):
        if torch.rand(1).item() < p:
            image = transforms.functional.hflip(image)
            # Flip landmarks horizontally
            landmarks[:, 0] = image.width - landmarks[:, 0]
        return image, landmarks

    def rotate(self, image, landmarks, degrees):
        angle = transforms.RandomRotation.get_params(degrees)
        image = transforms.functional.rotate(image, angle)

        # Rotate landmarks
        cx, cy = image.width / 2, image.height / 2  # Center of rotation
        angle_rad = -np.radians(angle)  # Negative for clockwise rotation
        cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)

        # Translate landmarks to origin, rotate, and translate back
        landmarks -= np.array([cx, cy])
        rotated_landmarks = np.zeros_like(landmarks)
        rotated_landmarks[:, 0] = landmarks[:, 0] * cos_theta - landmarks[:, 1] * sin_theta
        rotated_landmarks[:, 1] = landmarks[:, 0] * sin_theta + landmarks[:, 1] * cos_theta
        rotated_landmarks += np.array([cx, cy])
        return image, rotated_landmarks

    def random_resized_crop(self, image, landmarks, size, scale):
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=(3./4., 4./3.))
        image = transforms.functional.resized_crop(image, i, j, h, w, size)

        # Adjust landmarks for the crop
        landmarks[:, 0] -= j
        landmarks[:, 1] -= i
        crop_width, crop_height = w, h
        resize_width, resize_height = size

        # Scale landmarks to the new resized image
        landmarks[:, 0] *= resize_width / crop_width
        landmarks[:, 1] *= resize_height / crop_height
        return image, landmarks

# Root directory for the entire project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Directory to save detected faces for KMUFED, KDEF, FER2013 (united for both clean and adversarially trained results)
DETECTED_FACES_DIR = os.path.join(BASE_DIR, 'detected_faces')
KMUFED_FACE_DIR = os.path.join(DETECTED_FACES_DIR, 'kmufed')
KDEF_FACE_DIR = os.path.join(DETECTED_FACES_DIR, 'kdef')
FER2013_FACE_DIR = os.path.join(DETECTED_FACES_DIR, 'fer2013')

DATALOADERS_PATH = os.path.join(BASE_DIR, 'dataloaders')
DATASET_SUMMARY_PATH = os.path.join(BASE_DIR, 'dataset_summary')
# 2. Directory to save the plot of original vs detected face (united for both clean and adversarially trained results)
FACE_DETECTION_PLOT_DIR = os.path.join(BASE_DIR, 'face_detection_plots')

# 3. Directory to save training results for different models
TRAINING_RESULTS_DIR = os.path.join(BASE_DIR, 'training_results')
NORMAL_TRAINING_RESULTS_DIR = os.path.join(TRAINING_RESULTS_DIR, 'normal')
FGSM_TRAINING_RESULTS_DIR = os.path.join(TRAINING_RESULTS_DIR, 'fgsm')

# 4. Directory to save the model
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
NORMAL_MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, 'normal')
FGSM_MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, 'fgsm')

# 5. Directory to save the generated perturbed images for different attacks
PERTURBED_IMAGES_DIR = os.path.join(BASE_DIR, 'perturbed_images')

# 6. Directory to save the attack test results
ATTACK_RESULTS_DIR = os.path.join(BASE_DIR, 'attack_results')
NORMAL_ATTACK_RESULTS_DIR = os.path.join(ATTACK_RESULTS_DIR, 'normal')
FGSM_TRAINED_ATTACK_RESULTS_DIR = os.path.join(ATTACK_RESULTS_DIR, 'fgsm')

# Create all necessary directories
for directory in [
    DETECTED_FACES_DIR, KMUFED_FACE_DIR, KDEF_FACE_DIR, FER2013_FACE_DIR, DATALOADERS_PATH,DATASET_SUMMARY_PATH, FACE_DETECTION_PLOT_DIR, 
    NORMAL_TRAINING_RESULTS_DIR, FGSM_TRAINING_RESULTS_DIR,
    NORMAL_MODEL_SAVE_DIR, FGSM_MODEL_SAVE_DIR, PERTURBED_IMAGES_DIR,
    NORMAL_ATTACK_RESULTS_DIR, FGSM_TRAINED_ATTACK_RESULTS_DIR
]:
    ensure_dir_exists(directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
dataset_names = ['kmufed', 'kdef', 'fer2013']
model_names = ["ResNet18"]
num_classes = len(class_labels)
num_epochs = 50

transform = TransformWithLandmarks(
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
)