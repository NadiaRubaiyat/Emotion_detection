import torch
import time
from setup_training_grayscale import FGSM_MODEL_SAVE_DIR,FGSM_TRAINED_ATTACK_RESULTS_DIR,FGSM_TRAINING_RESULTS_DIR,FACE_DETECTION_PLOT_DIR ,device, num_classes, model_names, dataset_names, ensure_dir_exists, class_labels, summarize_results, num_epochs
from datasets_grayscale import KDEFDataset, FER2013Dataset, KMUFEDDataset
from dataloaders_grayscale import create_datasets, visualize_data, create_dataloaders
from attacks_grayscale import generate_adversarial_examples, run_attack
from utils_grayscale import initialize_model, train_model, evaluate_model, train_model_fgsm
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy
from torch.utils.data import DataLoader
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


dataset_info = {
    "kdef": {"path": r'C:\Users\Nadia\Desktop\Implementation\KDEF_and_AKDEF\KDEF_and_AKDEF\KDEF', "class": KDEFDataset},
    "kmufed": {"path": r'C:\Users\Nadia\Desktop\Implementation\KMU-FED', "class": KMUFEDDataset},
    "fer2013": {"path": r'C:\Users\Nadia\Desktop\Implementation\fer2013\fer2013.csv', "class": FER2013Dataset}
}

for dataset in dataset_names:
    dataset_instance = create_datasets(dataset_info[dataset]['path'], dataset_info[dataset]['class'], dataset)
    # # for idx in range(3):  # Test the first three samples
    # #     image, label, landmarks = dataset_instance[idx]
    # #     print(f"Image Shape: {image.shape}")
    # #     print(f"Label: {label}")
    # #     print(f"Landmarks: {landmarks}")
    for i in range(1):
    # Get image, label, and landmarks
        image, label, landmarks = dataset_instance[i]
        
        print("Normalized landmarks:", landmarks)  # Print the normalized landmarks

        # Convert normalized landmarks to pixel coordinates
        height, width = image.shape[1], image.shape[2]  # Get image dimensions (H, W)
        denormalized_landmarks = landmarks.clone()  # Clone landmarks to avoid modifying original
        denormalized_landmarks[:, 0] *= width  # Scale x-coordinates
        denormalized_landmarks[:, 1] *= height  # Scale y-coordinates

        print("Denormalized landmarks:", denormalized_landmarks)  # Debug the scaled landmarks

        # Visualize the image
        plt.imshow(image.permute(1, 2, 0).numpy(), cmap='gray')  # Convert tensor to image
        for point in denormalized_landmarks:
            plt.scatter(point[0], point[1], s=10, c='red')  # Plot landmarks in pixel space
        plt.title(f"Label: {label}")
        plt.show()

    # visualize_data(dataset_instance, dataset, FACE_DETECTION_PLOT_DIR)
    # train_loader, val_loader, test_loader = create_dataloaders(dataset_instance, dataset)

    # all_test_accuracies_fgsm = {}
    # all_test_reports_fgsm = {}
    # training_times_fgsm = {}

    # for model_name in model_names:
    #     result_file = f"{FGSM_TRAINING_RESULTS_DIR}/{model_name}"
    #     ensure_dir_exists(result_file)
    #     model = initialize_model(model_name, num_classes)
    #     model = model.to(device)

    #     print(f"Training {model_name} on data from {dataset}...")
    #     start_time = time.time()
    #     fgsm_image_dir=f'{FGSM_TRAINING_RESULTS_DIR}/{dataset}/{model_name}_adversarial_images'
    #     model = train_model_fgsm(model, train_loader, val_loader, model_name, dataset, result_file, fgsm_image_dir, epsilon=0.5/255, alpha=0.5, num_epochs=num_epochs)
    #     end_time = time.time()
    #     training_time = end_time - start_time
    #     training_times_fgsm[model_name] = training_time
    #     print(f"Training time for {model_name} on {dataset}: {training_time:.2f} seconds")

    #     print(f"Evaluating {model_name} on {dataset} dataset:")
    #     test_accuracy_clean, test_report_clean = evaluate_model(model, test_loader, dataset, model_name, result_file)

    #     all_test_accuracies_fgsm[model_name] = test_accuracy_clean
    #     all_test_reports_fgsm[model_name] = test_report_clean

    #     torch.save(model.state_dict(), f'{FGSM_MODEL_SAVE_DIR}/{dataset}_{model_name.lower()}.pth')
    #     print(f"Model saved in {FGSM_MODEL_SAVE_DIR}/{dataset}_{model_name.lower()}.pth")
    # summarize_results(all_test_accuracies_fgsm, all_test_reports_fgsm, training_times_fgsm, dataset, FGSM_TRAINING_RESULTS_DIR)
    
    # attacks = ["autoattack"]
    # result_file = f"{FGSM_TRAINED_ATTACK_RESULTS_DIR}/ResNet18"
    # attack_results = run_attack(test_loader, result_file, dataset, attacks, device, FGSM_MODEL_SAVE_DIR)

    torch.cuda.empty_cache()