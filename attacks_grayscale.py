import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import random
from tabulate import tabulate
import pandas as pd
from setup_training_grayscale import model_names, ensure_dir_exists, PERTURBED_IMAGES_DIR
from utils_grayscale import evaluate_model, load_model
from PIL import Image
import os

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def save_batch(image_tensor_batch, count, output_dir, prefix="image", batch_idx=0, epsilon=0):

    # Unnormalize the entire batch from [-1, 1] to [0, 1]
    unnormalized_images = image_tensor_batch * 0.5 + 0.5  # Undo normalization for entire batch

    # If the batch is grayscale, we need to handle it accordingly
    if unnormalized_images.shape[1] == 1:  # Grayscale image (C=1)
        unnormalized_images = unnormalized_images.squeeze(1)  # Remove the channel dimension, making it [B, H, W]

    # Convert the batch of tensors to NumPy arrays and save them
    for i in range(unnormalized_images.size(0)):  # Iterate over batch size
        unnormalized_image_np = unnormalized_images[i].cpu().numpy()  # Convert to NumPy [H, W]

        # Convert to a PIL image for saving
        unnormalized_image_pil = Image.fromarray((unnormalized_image_np * 255).astype(np.uint8), mode='L')

        # Save the image with a unique name based on count and batch index
        transformed_save_dir = f'{output_dir}'
        ensure_dir_exists(transformed_save_dir)  # Ensure directory exists
        unnormalized_image_pil.save(f"{transformed_save_dir}/{prefix}_{count + i}_{epsilon}.png")

def fgsm_attack(data, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = data + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image

def bim_attack(data, epsilon, alpha, num_iter, data_grad):
    """
    Performs the BIM attack by iteratively adding small perturbations.
    """
    perturbed_data = data.clone()

    for _ in range(num_iter):
        # Compute the sign of the gradient
        sign_data_grad = data_grad.sign()
        # Apply the perturbation
        perturbed_data = perturbed_data + alpha * sign_data_grad
        # Clip the perturbed image within epsilon-ball and maintain [0, 1] range
        perturbed_data = torch.clamp(perturbed_data, data - epsilon, data + epsilon)
        perturbed_data = torch.clamp(perturbed_data, -1, 1)
        perturbed_data.requires_grad_(True)
    return perturbed_data

def pgd_attack(model, data, target, epsilon, alpha, num_iter):
    """
    Performs the PGD attack by iteratively adding perturbations with gradient descent.
    """
    perturbed_data = data.clone().detach().requires_grad_(True)

    for _ in range(num_iter):
        # Forward pass
        output = model(perturbed_data)
        loss = F.cross_entropy(output, target)

        # Zero gradients and perform backpropagation
        model.zero_grad()
        loss.backward()

        # Apply perturbation based on gradient
        sign_data_grad = perturbed_data.grad.sign()
        perturbed_data = perturbed_data + alpha * sign_data_grad
        # Clip the perturbed image within epsilon-ball and maintain [0, 1] range
        perturbed_data = torch.clamp(perturbed_data, data - epsilon, data + epsilon)
        perturbed_data = torch.clamp(perturbed_data, -1, 1)

        # Detach and re-enable gradients for next iteration
        perturbed_data = perturbed_data.detach().requires_grad_(True)

    return perturbed_data


def generate_adversarial_examples(model, device, test_loader, epsilon, alpha, num_iter, attack_name, dataset_name):
    adversarial_examples = []
    adversarial_labels = []

    model.eval()
    adversarial_output_dir = f'adversarial_images_{dataset_name}'
    count = 0

    for idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        if attack_name == "fgsm":
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
        elif attack_name == "bim":
            perturbed_data = bim_attack(data, epsilon, alpha, num_iter, data_grad)
        elif attack_name == "pgd":
            perturbed_data = pgd_attack(model, data, target, epsilon, alpha, num_iter)
        else:
            raise ValueError(f"Attack {attack_name} is not supported.")

        # Detach to avoid retaining the computational graph
        perturbed_data = perturbed_data.detach()
        adversarial_examples.append(perturbed_data)
        adversarial_labels.append(target)

        save_batch(perturbed_data, count, f"{PERTURBED_IMAGES_DIR}/{dataset_name}/{attack_name}_images", prefix=f"adv_{attack_name}_{count}", batch_idx=idx, epsilon=epsilon)
        count += data.size(0)

    # Combine examples and create DataLoader
    adversarial_examples = torch.cat(adversarial_examples)
    adversarial_labels = torch.cat(adversarial_labels)
    adversarial_dataset = TensorDataset(adversarial_examples, adversarial_labels)
    adversarial_loader = DataLoader(adversarial_dataset, batch_size=64, shuffle=False)
    
    return adversarial_loader



def run_attack(test_loader, result_file, dataset_name, attacks, device, model_dir):
    epsilon_values = [0, 0.2/255, 0.4/255, 0.6/255, 0.8/255, 1/255]
    epsilon_labels = [f"{float(eps * 255)}/255" for eps in epsilon_values]  # Format for labeling
    alpha = 1
    num_iterations = 50

    attack_results = {}

    for attack_name in attacks:
        results = {"epsilon": epsilon_labels}  # Initialize results dictionary

        for model_name in model_names:
            results[f"{attack_name}_{model_name}"] = []  # Store accuracy per model
            model = load_model(dataset_name, model_name, model_dir)

            for epsilon in epsilon_values:
                adversarial_test_loader = generate_adversarial_examples(
                    model, device, test_loader, epsilon, alpha, num_iterations, attack_name, dataset_name
                )
                result_file_cm = f"{result_file}/Confusion_Matrix"
                ensure_dir_exists(result_file_cm)

                adv_accuracy, adv_report = evaluate_model(
                    model, adversarial_test_loader, dataset_name, model_name, result_file_cm
                )

                results[f"{attack_name}_{model_name}"].append(adv_accuracy * 100)  # Store accuracy percentage

            torch.cuda.empty_cache()

        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        attack_results[attack_name] = df_results  # Store per attack type

        csv_path = os.path.join(result_file, f"{attack_name}_attack_results_{dataset_name}.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"Saved attack results to {csv_path}")

        print(f"Results of the {attack_name} attack on {dataset_name} dataset:")
        print(df_results.to_string(index=False))

    # Generate a single plot with three subplots
    plot_attack_results_combined(attack_results, dataset_name, result_file)

    return attack_results


def plot_attack_results_combined(attack_results, dataset_name, result_file):
    """Generate a single figure with three subplots showing accuracy degradation for FGSM, BIM, and PGD attacks."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

    for idx, (attack_name, df_results) in enumerate(attack_results.items()):
        ax = axes[idx]

        for col in df_results.columns:
            if col != "epsilon":  # Avoid plotting epsilon values
                ax.plot(df_results["epsilon"], df_results[col], marker='o', label=col.replace(f"{attack_name}_", ""))

        ax.set_xlabel("Epsilon (Perturbation Strength)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{attack_name.upper()} Attack")
        ax.legend()
        ax.grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure
    plot_path = f"{result_file}/combined_attack_accuracy_drop_{dataset_name}.png"
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")

    plt.close()

