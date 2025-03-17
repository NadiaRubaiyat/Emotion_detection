import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.optim as optim 
import numpy as np
import random
from tabulate import tabulate
import pandas as pd
import os
from setup_training_grayscale import model_names, device, ensure_dir_exists, PERTURBED_IMAGES_DIR
from utils_grayscale import evaluate_model, load_model
from PIL import Image

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def save_batch(image_tensor_batch, count, output_dir, prefix="image", batch_idx=0):

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
        unnormalized_image_pil.save(f"{transformed_save_dir}/{prefix}_{count + i}.png")

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

        save_batch(perturbed_data, count, f"{PERTURBED_IMAGES_DIR}/{dataset_name}/{attack_name}_images", prefix=f"adv_{attack_name}_{count}", batch_idx=idx)
        count += data.size(0)

    # Combine examples and create DataLoader
    adversarial_examples = torch.cat(adversarial_examples)
    adversarial_labels = torch.cat(adversarial_labels)
    adversarial_dataset = TensorDataset(adversarial_examples, adversarial_labels)
    adversarial_loader = DataLoader(adversarial_dataset, batch_size=64, shuffle=False)
    
    return adversarial_loader


def run_attack(test_loader, result_file, dataset_name, attacks, device, model_dir):
    epsilon_values = [0, 0.2/255, 0.4/255, 0.6/255, 0.8/255, 1/255]
    alpha = 1
    num_iterations = 50

    for attack_name in attacks:
        results = {"epsilon": [], f"{attack_name}_ResNet18": [], f"{attack_name}_MobileNetV2": [], f"{attack_name}_EfficientNetB0": []}
        
        # Run FGSM, BIM, or PGD
        for i in range(len(epsilon_values)):
            numerator = float(epsilon_values[i] * 255)
            results["epsilon"].append(f"{numerator}/255")
        # Apply the attack on all the models
        for model_name in model_names:
            model = load_model(dataset_name, model_name, model_dir)
            dataname = f"{attack_name}_" + model_name
            for epsilon in epsilon_values:
                adversarial_test_loader = generate_adversarial_examples(model, device, test_loader, epsilon, alpha, num_iterations, attack_name, dataset_name)
                result_file_cm = f"{result_file}/Confusion_Matrix"
                ensure_dir_exists(result_file_cm)
                adv_accuracy, adv_report = evaluate_model(model, adversarial_test_loader, dataset_name, model_name, result_file_cm)
                results[dataname].append(f"{adv_accuracy * 100:.2f}%")
            torch.cuda.empty_cache()
        # Print the results
        print(f"Results of the {attack_name} attack on {dataset_name} dataset:")
        print(tabulate(results, headers="keys", tablefmt="pretty"))
        
        # Save results if needed
        torch.cuda.empty_cache()

    return results