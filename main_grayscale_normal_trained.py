import torch
import time
from setup_training_grayscale import NORMAL_MODEL_SAVE_DIR,device, num_classes, model_names, dataset_names, ensure_dir_exists, class_labels, summarize_results, num_epochs, FACE_DETECTION_PLOT_DIR, NORMAL_TRAINING_RESULTS_DIR
from datasets_grayscale import KDEFDataset, FER2013Dataset, KMUFEDDataset
from dataloaders_grayscale import create_datasets, visualize_data, create_dataloaders
from attacks_grayscale import generate_adversarial_examples, run_attack
from utils_grayscale import initialize_model, train_model, evaluate_model, train_model_fgsm
import pandas as pd
from tabulate import tabulate

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
    
    #visualize_data(dataset_instance, dataset, FACE_DETECTION_PLOT_DIR)
    train_loader, val_loader, test_loader = create_dataloaders(dataset_instance, dataset)

    all_test_accuracies_clean = {}
    all_test_reports_clean = {}
    training_times_clean = {}

    for model_name in model_names:
        result_file = f"{NORMAL_TRAINING_RESULTS_DIR}/{model_name}"
        ensure_dir_exists(result_file)
        model = initialize_model(model_name, num_classes)
        model = model.to(device)

        # Step 1: Clean training
        print(f"Training {model_name} on data from {dataset}...")
        start_time = time.time()

        model = train_model(model, train_loader, val_loader, model_name, dataset, result_file)
        end_time = time.time()
        training_time = end_time - start_time
        training_times_clean[model_name] = training_time
        print(f"Training time for {model_name} on {dataset}: {training_time:.2f} seconds")

        print(f"Evaluating {model_name} on {dataset} dataset:")
        test_accuracy_clean, test_report_clean = evaluate_model(model, test_loader, dataset, model_name, result_file)

        all_test_accuracies_clean[model_name] = test_accuracy_clean
        all_test_reports_clean[model_name] = test_report_clean

        torch.save(model.state_dict(), f'{NORMAL_MODEL_SAVE_DIR}/{dataset}_{model_name.lower()}.pth')

    summarize_results(all_test_accuracies_clean, all_test_reports_clean, training_times_clean, dataset, NORMAL_TRAINING_RESULTS_DIR)
    
    attacks = ["fgsm","bim","pgd"]
    attack_results = run_attack(test_loader, result_file, dataset, attacks, device, NORMAL_MODEL_SAVE_DIR)

    torch.cuda.empty_cache()