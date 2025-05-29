import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from utils_grayscale import load_model, evaluate_model
from attacks_grayscale import generate_adversarial_examples
from setup_training_grayscale import model_names, device

# === Configuration ===
datasets = ["kmufed", "kdef", "fer2013"]
model_variants = {
    "Clean": r"C:\Users\Nadia\Desktop\Implementation\Implement15\clean_without_mask\saved_models",
    "AttentionMasked": r"C:\Users\Nadia\Desktop\Implementation\Implement15\clean\saved_models"
}
epsilon = 0.5 / 255  # FGSM attack strength
dataloader_base = r"C:\Users\Nadia\Desktop\Implementation\Implement15\dataloaders"

# === Output Configuration ===
output_dir = r"C:\Users\Nadia\Desktop\Implementation\Implement15\new_test"
os.makedirs(output_dir, exist_ok=True)
save_csv_path = os.path.join(output_dir, "fgsm_cross_attack_comparison.csv")

results = []

for dataset in datasets:
    # Load test dataset splits saved as .pth files
    dataloader_dict_clean = torch.load(os.path.join(dataloader_base, f"{dataset}_dataloader_clean.pth"))
    dataloader_dict_masked = torch.load(os.path.join(dataloader_base, f"{dataset}_dataloader_masked.pth"))

    # Create actual test DataLoaders
    dataloaders = {
        "Clean": DataLoader(dataloader_dict_clean["test_dataset"], batch_size=64, shuffle=False),
        "AttentionMasked": DataLoader(dataloader_dict_masked["test_dataset"], batch_size=64, shuffle=False)
    }

    for model_name in model_names:
        print(f"\n=== Dataset: {dataset.upper()} | Model: {model_name} ===")

        # Load trained models (clean and attention-masked)
        models = {}
        for variant, model_path in model_variants.items():
            model = load_model(dataset, model_name, model_path).to(device)
            model.eval()
            models[variant] = model

        # Generate adversarial examples using each variant
        adversarial_sets = {}
        for source_variant in ["Clean", "AttentionMasked"]:
            print(f"Generating adversarial examples using {source_variant} model...")
            adv_loader = generate_adversarial_examples(
                model=models[source_variant],
                device=device,
                test_loader=dataloaders[source_variant],
                epsilon=epsilon,
                alpha=1,
                num_iter=1,
                attack_name="fgsm",
                dataset_name=dataset
            )
            adversarial_sets[source_variant] = adv_loader

        # Evaluate each model on each adversarial set
        for source_variant, adv_loader in adversarial_sets.items():
            for target_variant, target_model in models.items():
                acc, _ = evaluate_model(target_model, adv_loader, dataset, model_name, result_file=r"C:\Users\Nadia\Desktop\Implementation\Implement15\new_test")
                results.append({
                    "Dataset": dataset.upper(),
                    "Model": model_name,
                    "Adversarial From": source_variant,
                    "Evaluated Model": target_variant,
                    "Accuracy": acc * 100
                })

# Save final results
df = pd.DataFrame(results)
df.to_csv(save_csv_path, index=False)
print(f"\n✅ Results saved to: {save_csv_path}")
print(df)
