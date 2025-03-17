import torch
from torchvision import transforms
import os
from tabulate import tabulate
import pandas as pd

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

# Root directory for the entire project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAL_MAP_DIR = os.path.join(BASE_DIR, 'saliency_map')
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
    DETECTED_FACES_DIR, KMUFED_FACE_DIR, KDEF_FACE_DIR, FER2013_FACE_DIR, DATALOADERS_PATH, DATASET_SUMMARY_PATH, FACE_DETECTION_PLOT_DIR, 
    NORMAL_TRAINING_RESULTS_DIR, FGSM_TRAINING_RESULTS_DIR,
    NORMAL_MODEL_SAVE_DIR, FGSM_MODEL_SAVE_DIR, PERTURBED_IMAGES_DIR,
    NORMAL_ATTACK_RESULTS_DIR, FGSM_TRAINED_ATTACK_RESULTS_DIR
]:
    ensure_dir_exists(directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
dataset_names = ['kmufed','kdef','fer2013']
model_names = ["ResNet18","MobileNetV2","EfficientNetB0"]
num_classes = len(class_labels)
num_epochs = 50

# Updated Transformation Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)], p=0.5),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])