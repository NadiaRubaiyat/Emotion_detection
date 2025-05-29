import os
import time
import torch
from setup_training_grayscale import plot_model_metrics, SAL_MAP_DIR, NORMAL_MODEL_SAVE_DIR, NORMAL_MODEL_SAVE_DIR,NORMAL_ATTACK_RESULTS_DIR,NORMAL_TRAINING_RESULTS_DIR,FACE_DETECTION_PLOT_DIR ,device, model_names, dataset_names, ensure_dir_exists, summarize_results, num_epochs
from datasets_grayscale import KDEFDataset, FER2013Dataset, KMUFEDDataset
from dataloaders_grayscale import create_datasets, visualize_data, create_dataloaders
from attacks_grayscale import run_attack
from utils_grayscale import load_model,load_training_data, initialize_model,  evaluate_model, train_model, generate_saliency_map, visualize_saliency_map, plot_combined_losses

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

    all_test_accuracies = {}
    all_test_reports = {}
    training_times = {}
    loss_dict = {}
    for model_name in model_names:
        result_file = f"{NORMAL_TRAINING_RESULTS_DIR}/{model_name}"
        ensure_dir_exists(result_file)
        if dataset == "kmufed":
            num_classes = 6
        else:
            num_classes = 7
        model = initialize_model(model_name, num_classes).to(device)

        # print(f"Training {model_name} on data from {dataset}...")
        # start_time = time.time()
        # model = train_model(model, train_loader, val_loader, model_name, dataset, result_file, loss_dict)
        # end_time = time.time()
        # training_time = end_time - start_time
        # training_times[model_name] = training_time
        # print(f"Training time for {model_name} on {dataset}: {training_time:.2f} seconds")

        # print(f"Evaluating {model_name} on {dataset} dataset:")
        # test_accuracy_clean, test_report_clean = evaluate_model(model, test_loader, dataset, model_name, result_file)
        # print(f"Clean Test Accuracy for {model_name} on {dataset}: {test_accuracy_clean:.4f}")

        # all_test_accuracies[model_name] = test_accuracy_clean
        # all_test_reports[model_name] = test_report_clean

        # torch.save(model.state_dict(), f'{NORMAL_MODEL_SAVE_DIR}/{dataset}_{model_name.lower()}.pth')
        # print(f"Model saved in {NORMAL_MODEL_SAVE_DIR}/{dataset}_{model_name.lower()}.pth")
        model_dir = r"C:\Users\Nadia\Desktop\Implementation\Implement15\fgsm\saved_models"
        model = load_model(dataset, model_name, model_dir)
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
        indices = [10060, 10086, 10110, 10135, 10161]
        
        saliency_path = os.path.join(saliency_dir, f"saliency_map_combined_{dataset}_{model_name}.png")
        visualize_saliency_map(saliency_data, saliency_path)

#     df = summarize_results(all_test_accuracies, all_test_reports, training_times, dataset, NORMAL_TRAINING_RESULTS_DIR)
#     plot_model_metrics(df, dataset,NORMAL_TRAINING_RESULTS_DIR)
#     # plot_combined_losses(loss_dict, num_epochs, dataset, NORMAL_TRAINING_RESULTS_DIR)
    
#     attacks = ["fgsm","bim","pgd"]
#     result_file = os.path.join(NORMAL_ATTACK_RESULTS_DIR, "")
#     print(result_file)
#     attack_results = run_attack(test_loader, result_file, dataset, attacks, device, NORMAL_MODEL_SAVE_DIR)
    
#     # del train_loader, val_loader, test_loader, dataset_instance, model
#     torch.cuda.empty_cache()

# datasets = ["KMUFED", "KDEF", "FER2013"]
# models = ["ResNet18", "EfficientNetB0", "MobileNetV2"]

# # Base directory for training results
# base_path = r"C:\Users\Nadia\Desktop\Implementation\Implement15\clean\training_results"

# # Dictionary to store loss data for all datasets
# all_losses = {}

# # Loop through each dataset and model
# for dataset in datasets:
#     losses_dict = {}  # Temporary dictionary for storing losses for current dataset

#     for model in models:
#         csv_path = rf"{base_path}\{model}\training_validation_accuracies_{model}_{dataset.lower()}.csv"

#         # Load data into dictionary
#         losses_dict[model] = load_training_data(csv_path)

#     # Store dataset-specific losses in all_losses dictionary
#     all_losses[dataset] = losses_dict

# plot_combined_losses(all_losses, num_epochs=50, result_file=base_path)