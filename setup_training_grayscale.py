import torch
from torchvision import transforms
import os
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt

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
    summary_df.to_csv(f"{path}/model_results_{dataset}.csv", index=False)
    print(f"\n{dataset} {path.capitalize()} Results Summary:")
    print(tabulate(summary_rows, headers="keys", tablefmt="pretty"))
    return summary_df

def plot_model_metrics(df_summary, dataset_name, result_file):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    models = df_summary["Model"]
    colors = ['#4E79A7', '#F28E2B', '#76B7B2', '#E15759']  # Optional: model colors

    x = range(len(models))
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        values = df_summary[metric].values * 100  # Convert to percentage
        positions = [pos + i * bar_width for pos in x]
        ax.bar(positions, values, bar_width, label=metric, color=colors[i % len(colors)])

        # Add values on top
        for pos, val in zip(positions, values):
            ax.text(pos, val + 1, f"{val:.1f}%", ha='center', fontsize=9)

    ax.set_xlabel("Models")
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Performance Metrics on {dataset_name}")
    ax.set_xticks([r + 1.5 * bar_width for r in x])
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = f"{result_file}/{dataset_name}_precision_recall_f1_accuracy.png"
    plt.savefig(plot_path)

# Root directory for the entire project
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'fgsm')

SAL_MAP_DIR = os.path.join(BASE_DIR, 'saliency_map')
# 1. Directory to save detected faces for KMUFED, KDEF, FER2013 (united for both clean and adversarially trained results)
DETECTED_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'detected_faces')
KMUFED_FACE_DIR = os.path.join(DETECTED_FACES_DIR, 'kmufed_images')
KDEF_FACE_DIR = os.path.join(DETECTED_FACES_DIR, 'kdef_images')
FER2013_FACE_DIR = os.path.join(DETECTED_FACES_DIR, 'fer2013_images')

DATALOADERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'dataloaders')
DATASET_SUMMARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_summary')
# 2. Directory to save the plot of original vs detected face (united for both clean and adversarially trained results)
FACE_DETECTION_PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_detection_plots')

# 3. Directory to save training results for different models
TRAINING_RESULTS_DIR = os.path.join(BASE_DIR, 'training_results')
NORMAL_TRAINING_RESULTS_DIR = TRAINING_RESULTS_DIR
FGSM_TRAINING_RESULTS_DIR = TRAINING_RESULTS_DIR

# 4. Directory to save the model
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
NORMAL_MODEL_SAVE_DIR = MODEL_SAVE_DIR
FGSM_MODEL_SAVE_DIR = MODEL_SAVE_DIR

# 5. Directory to save the generated perturbed images for different attacks
PERTURBED_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'perturbed_images')

# 6. Directory to save the attack test results
ATTACK_RESULTS_DIR = os.path.join(BASE_DIR, 'attack_results')
NORMAL_ATTACK_RESULTS_DIR = ATTACK_RESULTS_DIR
FGSM_TRAINED_ATTACK_RESULTS_DIR = ATTACK_RESULTS_DIR

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
dataset_names = ['fer2013']
model_names = ["ResNet18","EfficientNetB0","MobileNetV2"]
num_classes = len(class_labels)
num_epochs = 70

# Updated Transformation Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)], p=0.5),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])