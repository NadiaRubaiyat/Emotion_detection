import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import pandas as pd
import os
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights, EfficientNet_B0_Weights
import torchvision.models as models
import torch.nn as nn
from setup_training_grayscale import num_epochs, device, ensure_dir_exists
import torchvision.transforms as transforms
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
import cv2
from matplotlib.colors import Normalize

def initialize_model(model_name, num_classes, weights=True):
    if model_name == "ResNet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if weights else None)
        num_ftrs = model.fc.in_features
        
        # Modify the first convolutional layer to accept single-channel grayscale input
        model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
        
        # Modify the final fully connected layer for your classes
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )

    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if weights else None)
        num_ftrs = model.classifier[-1].in_features
        
        # Modify the first convolutional layer to accept single-channel grayscale input
        model.features[0][0] = nn.Conv2d(1, model.features[0][0].out_channels, kernel_size=model.features[0][0].kernel_size, stride=model.features[0][0].stride, padding=model.features[0][0].padding, bias=False)
        
        # Modify the final classifier for your classes
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )
        
    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if weights else None)
        num_ftrs = model.classifier[1].in_features
        
        # Modify the first convolutional layer to accept single-channel grayscale input
        model.features[0][0] = nn.Conv2d(1, model.features[0][0].out_channels, kernel_size=model.features[0][0].kernel_size, stride=model.features[0][0].stride, padding=model.features[0][0].padding, bias=False)
        
        # Modify the final classifier for your classes
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


def train_model(model, train_loader, val_loader, model_name, dataset_name, result_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_accuracy = 0.0
    best_model_weights = None

    # Define the loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    weight_decay=1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []

        for data, labels in train_loader:
            optimizer.zero_grad()
            data, labels = data.to(device), labels.to(device)      

            # Forward pass on combined data
            data.requires_grad = True
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item() * data.size(0)

            # Track predictions for accuracy
            _, preds = torch.max(outputs, 1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())
        
        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        running_loss = 0.0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_accuracies.append(val_accuracy)

        # Adjust learning rate based on validation loss
        scheduler.step(epoch_val_loss)

        # Print training and validation metrics every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, "
                  f"Train Accuracy: {train_accuracies[-1] * 100:.2f}%, "
                  f"Validation Loss: {val_losses[-1]:.4f}, "
                  f"Validation Accuracy: {val_accuracies[-1] * 100:.2f}%")

        # Save best model weights
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_weights = model.state_dict()

    # Load the best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Save training/validation statistics to CSV
    epoch_data = pd.DataFrame({
        "Epoch": list(range(1, num_epochs + 1)),
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
        "Train Accuracy": [acc * 100 for acc in train_accuracies],
        "Validation Accuracy": [acc * 100 for acc in val_accuracies]
    })
    epoch_data.to_csv(f"{result_file}/training_validation_accuracies_{model_name}_{dataset_name}_{timestamp}.csv", index=False)
    torch.cuda.empty_cache()
    return model

def train_model_fgsm(model, train_loader, val_loader, model_name, dataset_name, result_file, fgsm_image_dir, epsilon=0.5/255, alpha=0.5, num_epochs=50):
    ensure_dir_exists(fgsm_image_dir)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = 0.0
    best_model_weights = None

    criterion = torch.nn.CrossEntropyLoss()
    weight_decay = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        batch_index = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False)

        for data, labels in train_loader_tqdm:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            data.requires_grad = True
            clean_outputs = model(data)
            clean_loss = criterion(clean_outputs, labels)
            clean_loss.backward(retain_graph=True)
            data_grad = data.grad.data
            perturbed_data = fgsm_attack(data, epsilon, data_grad, batch_index, fgsm_image_dir)

            perturbed_outputs = model(perturbed_data)
            adv_loss = criterion(perturbed_outputs, labels)
            
            total_loss = alpha * clean_loss + (1 - alpha) * adv_loss
            
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * data.size(0)
            _, preds = torch.max(clean_outputs, 1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())
            batch_index += 1

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_accuracies.append(train_accuracy)

        model.eval()
        running_val_loss = 0.0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_accuracies.append(val_accuracy)

        scheduler.step(epoch_val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, "
                  f"Train Accuracy: {train_accuracy * 100:.2f}%, "
                  f"Validation Loss: {epoch_val_loss:.4f}, "
                  f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_weights = model.state_dict()

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    epoch_data = pd.DataFrame({
        "Epoch": list(range(1, num_epochs + 1)),
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
        "Train Accuracy": [acc * 100 for acc in train_accuracies],
        "Validation Accuracy": [acc * 100 for acc in val_accuracies]
    })
    epoch_data.to_csv(f"{result_file}/training_validation_accuracies_{model_name}_{dataset_name}.csv", index=False)
    torch.cuda.empty_cache()
    return model

def train_model_pgd(model, train_loader, val_loader, model_name, dataset_name, result_file, fgsm_image_dir, epsilon=0.5/255, alpha=0.5, num_epochs=50):
    ensure_dir_exists(fgsm_image_dir)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = 0.0
    best_model_weights = None

    criterion = torch.nn.CrossEntropyLoss()
    weight_decay = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        batch_index = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=False)

        for data, labels in train_loader_tqdm:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            data.requires_grad = True
            clean_outputs = model(data)
            clean_loss = criterion(clean_outputs, labels)
            clean_loss.backward(retain_graph=True)
            data_grad = data.grad.data
            perturbed_data = pgd_attack(data, epsilon, data_grad, batch_index, fgsm_image_dir)

            perturbed_outputs = model(perturbed_data)
            adv_loss = criterion(perturbed_outputs, labels)
            
            total_loss = alpha * clean_loss + (1 - alpha) * adv_loss
            
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * data.size(0)
            _, preds = torch.max(clean_outputs, 1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())
            batch_index += 1

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_accuracies.append(train_accuracy)

        model.eval()
        running_val_loss = 0.0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_accuracies.append(val_accuracy)

        scheduler.step(epoch_val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, "
                  f"Train Accuracy: {train_accuracy * 100:.2f}%, "
                  f"Validation Loss: {epoch_val_loss:.4f}, "
                  f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_weights = model.state_dict()

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    epoch_data = pd.DataFrame({
        "Epoch": list(range(1, num_epochs + 1)),
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
        "Train Accuracy": [acc * 100 for acc in train_accuracies],
        "Validation Accuracy": [acc * 100 for acc in val_accuracies]
    })
    epoch_data.to_csv(f"{result_file}/training_validation_accuracies_{model_name}_{dataset_name}.csv", index=False)
    torch.cuda.empty_cache()
    return model

def evaluate_model(model, test_loader, dataset_name, model_name, result_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.to(device)
    all_test_labels = []
    all_test_predictions = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_test_labels.extend(labels.cpu().numpy())
            all_test_predictions.extend(predicted.cpu().numpy())

    all_test_labels = np.array(all_test_labels)
    all_test_predictions = np.array(all_test_predictions)
    if "kmufed" in dataset_name.lower():
        count = 6
        class_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    else:
        count = 7
        class_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
    
    target_names = [class_labels[i] for i in range(count)]

    # Calculate confusion matrix and normalize by row (i.e., by true label counts)
    
    cm_test = confusion_matrix(all_test_labels, all_test_predictions, labels=np.arange(count), normalize='true')

    # Display confusion matrix
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=target_names)
    disp_test.plot(cmap=plt.cm.Blues, values_format='.2f')
    plt.title(f'Confusion Matrix for Test Set on {dataset_name} - {model_name}')
    plt.savefig(f"{result_file}/Confusion_matrix_{dataset_name}_{model_name}_{timestamp}")
    plt.close()

    # Calculate and print precision, recall, and F1-score for test set
    test_report = classification_report(all_test_labels, all_test_predictions, target_names=target_names, labels=np.arange(count), output_dict=True, zero_division=1)
    
    # Calculate and print accuracy for test set
    test_accuracy = accuracy_score(all_test_predictions, all_test_labels)

    print(f"Test accuracy: {test_accuracy:.4f}")

    return test_accuracy, test_report

def generate_saliency_map(model, image, label):
    # Set model to evaluation mode
    model.eval()
    
    # Make sure the image requires gradients
    image = image.unsqueeze(0).to(device)  # Add a batch dimension and move to device
    image.requires_grad_()  # Enable gradient computation on the image
    
    # Forward pass
    output = model(image)

    # Get the index of the class with the highest score
    pred_class = output.argmax(dim=1).item()

    # Backward pass for the highest score
    model.zero_grad()  # Clear any existing gradients
    output[0, pred_class].backward()  # Compute the gradient of the top class

    # Get the gradient of the input image
    saliency = image.grad.data.abs().squeeze().cpu().numpy()

    # Aggregate across color channels if necessary
    if saliency.ndim == 3:
        saliency = np.max(saliency, axis=0)  # Take maximum across channels for visualization

    return saliency

def visualize_saliency_map(image, saliency, save_path):
    image = image.squeeze().cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]

    # Normalize the saliency map for overlay
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())  # Normalize to [0, 1]

    # Create a figure for overlay
    plt.figure(figsize=(8, 8))

    # Display the grayscale image
    plt.imshow(image, cmap='gray', interpolation='nearest')

    # # Overlay the saliency map using a colormap and transparency (alpha)
    plt.imshow(
        saliency,
        cmap='jet',
        norm=Normalize(vmin=0, vmax=1),
        alpha=0.7  # Adjust transparency
    )

    # Remove axes for clean visualization
    plt.axis('off')

    # Save the resulting overlay
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def load_model(dataset_name, model_name, directory):
    if "kmufed" in dataset_name.lower():
        num_classes = 6
    else: 
        num_classes = 7
    os.makedirs(directory, exist_ok=True) 
    # Initialize the model
    model = initialize_model(model_name, num_classes) 
    # Path to the saved model weights
    model_path = f'{directory}/{dataset_name}_{model_name.lower()}.pth'
    try:
        # Load the saved state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model {model_name} loaded successfully from {model_path}.")
    except FileNotFoundError:
        raise FileNotFoundError(f"No saved model found at {model_path}. Please ensure the model is trained and saved.")
    except RuntimeError as e:
        raise RuntimeError(f"Error loading model: {e}. Ensure the saved model matches the architecture.")
    # Move model to the appropriate device
    model = model.to(device)
    return model

def plot_losses(train_losses, val_losses, num_epochs, model_name, dataset_name, result_file):
    if "kmufed" in dataset_name.lower():
        count = 6
    else:
        count = 7
    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(10, count))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title(f"Training and Validation Loss for {model_name} on {dataset_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = f"{result_file}/loss_plot_{model_name}_{dataset_name}.png"
    plt.savefig(plot_path)
    plt.close()

def save_perturbed_image(perturbed_image, index, output_dir, count):
    un_normalized_image = (perturbed_image + 1) / 2 
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(un_normalized_image.cpu())  
    image_path = os.path.join(output_dir, f"perturbed_image_{index}_{count}.png")
    pil_image.save(image_path)


def fgsm_attack(data, epsilon, data_grad, batch_index, output_dir):
    sign_data_grad = data_grad.sign()
    perturbed_images = data + epsilon * sign_data_grad
    perturbed_images = torch.clamp(perturbed_images, -1, 1)
    for i in range(perturbed_images.size(0)):  
        perturbed_image = perturbed_images[i]  
        save_perturbed_image(perturbed_image.squeeze(), batch_index + i, output_dir, i)
    return perturbed_images

def pgd_attack(model, data, target, epsilon, alpha, num_iter):
    perturbed_data = data.clone().detach().requires_grad_(True)
    for _ in range(num_iter):
        output = model(perturbed_data)
        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()
        sign_data_grad = perturbed_data.grad.sign()
        perturbed_data = perturbed_data + alpha * sign_data_grad
        perturbed_data = torch.clamp(perturbed_data, data - epsilon, data + epsilon)
        perturbed_data = torch.clamp(perturbed_data, -1, 1)
        perturbed_data = perturbed_data.detach().requires_grad_(True)
    return perturbed_data