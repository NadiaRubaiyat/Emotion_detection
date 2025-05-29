import os
import time
import torch
from setup_training_grayscale import plot_model_metrics, SAL_MAP_DIR, FGSM_MODEL_SAVE_DIR,FGSM_TRAINED_ATTACK_RESULTS_DIR,FGSM_TRAINING_RESULTS_DIR,FACE_DETECTION_PLOT_DIR ,device, model_names, dataset_names, ensure_dir_exists, summarize_results, num_epochs
from datasets_grayscale import KDEFDataset, FER2013Dataset, KMUFEDDataset
from dataloaders_grayscale import create_datasets, visualize_data, create_dataloaders
from attacks_grayscale import run_attack
from utils_grayscale import load_training_data, initialize_model,  evaluate_model, train_model_fgsm, generate_saliency_map, plot_combined_losses,visualize_saliency_map

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

    visualize_data(dataset_instance, dataset, FACE_DETECTION_PLOT_DIR)
    train_loader, val_loader, test_loader = create_dataloaders(dataset_instance, dataset)

    all_test_accuracies_fgsm = {}
    all_test_reports_fgsm = {}
    training_times_fgsm = {}
    loss_dict = {}
    for model_name in model_names:
        result_file = f"{FGSM_TRAINING_RESULTS_DIR}/{model_name}"
        ensure_dir_exists(result_file)
        if dataset == "kmufed":
            num_classes = 6
        else:
            num_classes = 7
        model = initialize_model(model_name, num_classes).to(device)

        print(f"Training {model_name} on data from {dataset}...")
        start_time = time.time()
        fgsm_image_dir=f'{FGSM_TRAINING_RESULTS_DIR}/{dataset}/{model_name}_adversarial_images'
        model = train_model_fgsm(model, train_loader, val_loader, model_name, dataset, result_file, fgsm_image_dir, loss_dict, epsilon=0.5/255, alpha=0.5, num_epochs=num_epochs)
        end_time = time.time()
        training_time = end_time - start_time
        training_times_fgsm[model_name] = training_time
        print(f"Training time for {model_name} on {dataset}: {training_time:.2f} seconds")

        print(f"Evaluating {model_name} on {dataset} dataset:")
        test_accuracy_adtrain, test_report_adtrain = evaluate_model(model, test_loader, dataset, model_name, result_file)
        print(f"Clean Test Accuracy for Adversarially trained {model_name} on {dataset}: {test_accuracy_adtrain:.4f}")
        
        all_test_accuracies_fgsm[model_name] = test_accuracy_adtrain
        all_test_reports_fgsm[model_name] = test_report_adtrain

        torch.save(model.state_dict(), f'{FGSM_MODEL_SAVE_DIR}/{dataset}_{model_name.lower()}.pth')
        print(f"Model saved in {FGSM_MODEL_SAVE_DIR}/{dataset}_{model_name.lower()}.pth")

        saliency_dir = os.path.join(SAL_MAP_DIR, f"{dataset}_{model_name}")
        ensure_dir_exists(saliency_dir)

        print(f"Generating saliency maps for {model_name} on {dataset}...")

        # List to store image-saliency pairs
        saliency_data = []

        for i, (image, label) in enumerate(test_loader.dataset):
            if i >= 5:
                break
            saliency = generate_saliency_map(model, image, label)
            saliency_data.append((image, saliency))

        # Save all visualizations in one image
        saliency_path = os.path.join(saliency_dir, f"saliency_map_combined_{dataset}_{model_name}.png")
        visualize_saliency_map(saliency_data, saliency_path)

    df = summarize_results(all_test_accuracies_fgsm, all_test_reports_fgsm, training_times_fgsm, dataset, FGSM_TRAINING_RESULTS_DIR)
    # plot_combined_losses(loss_dict, num_epochs, dataset, result_file)
    plot_model_metrics(df, dataset,FGSM_TRAINING_RESULTS_DIR)

    attacks = ["fgsm","bim","pgd"]
    result_file = os.path.join(FGSM_TRAINED_ATTACK_RESULTS_DIR, "")
    print(result_file)
    attack_results = run_attack(test_loader, result_file, dataset, attacks, device, FGSM_MODEL_SAVE_DIR)
    
    # del train_loader, val_loader, test_loader, dataset_instance, model
    torch.cuda.empty_cache()

datasets = ["KMUFED", "KDEF", "FER2013"]
models = ["ResNet18", "EfficientNetB0", "MobileNetV2"]

# Base directory for training results
base_path = r"C:\Users\Nadia\Desktop\Implementation\Implement15\clean\training_results"

# Dictionary to store loss data for all datasets
all_losses = {}

# Loop through each dataset and model
for dataset in datasets:
    losses_dict = {}  # Temporary dictionary for storing losses for current dataset

    for model in models:
        csv_path = rf"{base_path}\{model}\training_validation_accuracies_{model}_{dataset.lower()}.csv"

        # Load data into dictionary
        losses_dict[model] = load_training_data(csv_path)

    # Store dataset-specific losses in all_losses dictionary
    all_losses[dataset] = losses_dict

# plot_combined_losses(all_losses, num_epochs=50, result_file=base_path)