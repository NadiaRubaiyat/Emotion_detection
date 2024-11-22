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
from torchvision.transforms import ToPILImage
from setup_training_grayscale import model_names, device, ensure_dir_exists, PERTURBED_IMAGES_DIR
from utils_grayscale import evaluate_model, load_model
from PIL import Image
from autoattack import AutoAttack
import foolbox as fb
import eagerpy as ep

# Set the random seed for Python's random module
random.seed(42)
# Set the random seed for numpy
np.random.seed(42)
# Set the random seed for PyTorch
torch.manual_seed(42)
# If using GPU, also set the seed for CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def deepfool_attack(model, images, labels):
    model.eval()  # Set the model to evaluation mode
    
    # Make sure images require gradients
    images = images.detach().clone().requires_grad_(True)
    
    # Wrap the model in Foolbox with appropriate bounds (ensure these bounds match your preprocessing)
    fmodel = fb.PyTorchModel(model, bounds=(-1, 1))

    # Convert data and target to EagerPy tensors, which Foolbox requires
    ep_images = ep.astensor(images)
    ep_labels = ep.astensor(labels)

    # Initialize the DeepFool attack
    attack = fb.attacks.LinfDeepFoolAttack()

    # Run the attack and catch any errors
    try:
        advs = attack(fmodel, ep_images, criterion=fb.criteria.Misclassification(ep_labels), epsilons=None)
        return advs.tensor  # Convert back to a standard PyTorch tensor
    except Exception as e:
        print(f"Error in DeepFool attack: {e}")
        return images  # Return original images if the attack fails


# def visualize_attack(model, device, test_loader, epsilon_values, result_file, dataset_name, alpha=0.001, num_iterations=1, attack_name="fgsm"):
#     adv_examples = []

#     #ensure_dir_exists(result_file)

#     for data, target in test_loader:
#         data = (data + 1) / 2
#         data, target = data.to(device), target.to(device)
#         data.requires_grad = True

#         output = model(data)

#         # Store the original image (before perturbation) for visualization
#         original_image = data[0].detach().cpu().numpy().squeeze()  # No need to unnormalize here, it's already [0, 1]

#         perturbed_images = []
#         for epsilon in epsilon_values:
#             loss = F.cross_entropy(output, target)
#             model.zero_grad()
#             loss.backward(retain_graph=True)
#             data_grad = data.grad.data

#             if attack_name == "fgsm":
#                 perturbed_data = fgsm_attack(data, epsilon, data_grad)
#             elif attack_name == "bim":
#                 perturbed_data = bim_attack(data, epsilon, alpha, num_iterations, data_grad)
#             elif attack_name == "pgd":
#                 perturbed_data = pgd_attack(model, data, target, epsilon, alpha, num_iterations)

#             # 3. Re-normalize the perturbed data from [0, 1] back to [-1, 1] for the model (if needed)
#             perturbed_data = (perturbed_data * 2) - 1
#             perturbed_image = perturbed_data[0].detach().cpu().numpy().squeeze()
#             perturbed_images.append(perturbed_image)
#         data.requires_grad_(False)
#         adv_examples.append((original_image, perturbed_images))
#         break  # Only visualize the first batch

#     # Visualize the original and perturbed images
#     for original, perturbed_images in adv_examples:
#         plt.figure(figsize=(15, 4))

#         # Original image
#         plt.subplot(1, len(epsilon_values) + 1, 1)
#         plt.title("Original Image")
#         plt.imshow(original, cmap='gray')  # Display as grayscale
#         plt.axis('off')

#         # Perturbed images
#         for i, perturbed in enumerate(perturbed_images):
#             plt.subplot(1, len(epsilon_values) + 1, i + 2)
#             plt.title(f"Epsilon = {epsilon_values[i]}")
#             plt.imshow(perturbed, cmap='gray')  # Display as grayscale
#             plt.axis('off')

#         # Save the visualized images to a file
        
#         plt.savefig(f"{result_file}/{attack_name}_attack_for_{dataset_name}")
#         # plt.show()
#         plt.close()

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

# Modified visualization to handle normalization

def fgsm_attack(data, epsilon, data_grad):
    """
    Performs the FGSM attack by adding epsilon * sign(data_grad) to the input data.
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = data + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    return perturbed_image

def run_autoattack(model, test_loader, epsilon, device, dataset_name):
    """
    Runs AutoAttack on the model and evaluates accuracy on adversarial examples.
    """
    # Set up AutoAttack with only the compatible untargeted attacks
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, device=device)
    adversary.attacks_to_run = ['apgd-ce', 'square']  # Exclude 'apgd-t' and 'fab-t'

    # Prepare lists to store results
    all_adv_preds = []
    all_labels = []
    
    model.eval()  # Set the model to evaluation mode

    # Loop through the test_loader and apply AutoAttack
    for data, target, _ in test_loader:
        data, target = data.to(device), target.to(device)
        
        # Generate adversarial examples using AutoAttack
        adv_data = adversary.run_standard_evaluation(data, target, bs=len(data))
        
        # Predict on adversarial examples
        with torch.no_grad():
            output = model(adv_data)
            _, preds = output.max(1)
            all_adv_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Calculate accuracy on adversarial examples
    correct = sum(int(p == l) for p, l in zip(all_adv_preds, all_labels))
    accuracy = 100 * correct / len(all_labels)
    print(f"Accuracy on AutoAttack adversarial examples: {accuracy:.2f}%")

    # Optionally, save results if required
    results_df = pd.DataFrame({
        "Adversarial Prediction": all_adv_preds,
        "True Label": all_labels
    })
    save_path = os.path.join(PERTURBED_IMAGES_DIR, f"{dataset_name}_autoattack_results.csv")
    results_df.to_csv(save_path, index=False)
    print(f"AutoAttack results saved to {save_path}")

    return accuracy


from autoattack import AutoAttack

def generate_adversarial_examples(model, device, test_loader, epsilon, alpha, num_iter, attack_name, dataset_name, task):
    adversarial_examples = []
    adversarial_labels = []

    model.eval()
    adversarial_output_dir = f'adversarial_images_{dataset_name}'
    count = 0

    for idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        # Check for AutoAttack
        if attack_name == "autoattack":
            # Initialize AutoAttack
            adversary = AutoAttack(model, norm='Linf', eps=epsilon, device=device)
            adv_data = adversary.run_standard_evaluation(data, target, bs=len(data))
            perturbed_data = adv_data  # Set perturbed_data directly to adv_data from AutoAttack
        else:
            # Handle FGSM, BIM, PGD, DeepFool as per your existing code
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
            elif attack_name == "deepfool":
                perturbed_data = deepfool_attack(model, data, target)
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
    num_iterations = 10

    for attack_name in attacks:
        results = {"epsilon": [], f"{attack_name}_ResNet18": [], f"{attack_name}_MobileNetV2": [], f"{attack_name}_EfficientNetB0": []}
        
        if attack_name.lower() == "autoattack":
            # Set epsilon for AutoAttack (typically 8/255 for robustness evaluation)
            epsilon = 8 / 255
            for model_name in model_names:
                model = load_model(dataset_name, model_name, model_dir)
                print(f"Running AutoAttack on {model_name} for {dataset_name} dataset...")
                
                # Run AutoAttack and get accuracy
                accuracy = run_autoattack(model, test_loader, epsilon, device, dataset_name)
                results[f"{attack_name}_{model_name}"].append(f"{accuracy:.2f}%")
        
        else:
            # Run FGSM, BIM, or PGD
            for i in range(len(epsilon_values)):
                results["epsilon"].append(f"0.{i*2}/255")

            # Apply the attack on all the models
            for model_name in model_names:
                model = load_model(dataset_name, model_name, model_dir)
                dataname = f"{attack_name}_" + model_name
                for epsilon in epsilon_values:
                    adversarial_test_loader = generate_adversarial_examples(model, device, test_loader, epsilon, alpha, num_iterations, attack_name, dataset_name, "test")
                    result_file_cm = f"{result_file}/Confusion_Matrix"
                    ensure_dir_exists(result_file_cm)
                    adv_accuracy, adv_report = evaluate_model(model, adversarial_test_loader, dataname, model_name, result_file_cm)
                    results[dataname].append(f"{adv_accuracy * 100:.2f}%")
        
        # Print the results
        print(f"Results of the {attack_name} attack on {dataset_name} dataset:")
        print(tabulate(results, headers="keys", tablefmt="pretty"))
        
        # Save results if needed
        torch.cuda.empty_cache()

    return results

