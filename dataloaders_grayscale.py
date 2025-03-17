import os
import pandas as pd
import torch
import numpy as np
import random
from setup_training_grayscale import transform, DATASET_SUMMARY_PATH,DATALOADERS_PATH, KMUFED_FACE_DIR, KDEF_FACE_DIR, FER2013_FACE_DIR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import Counter
from tabulate import tabulate
import matplotlib.pyplot as plt

# Set the random seed for Python's random module
random.seed(42)
# Set the random seed for numpy
np.random.seed(42)
# Set the random seed for PyTorch
torch.manual_seed(42)
# If using GPU, also set the seed for CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def create_datasets(dataset_path, dataset_class, dataset_name):
    if dataset_name == "fer2013":
        data_paths = dataset_path
        face_dir = FER2013_FACE_DIR
    elif dataset_name == "kmufed":
        # Collect image file paths
        face_dir = KMUFED_FACE_DIR
        data_paths = [os.path.join(dataset_path, fname) 
                    for fname in os.listdir(dataset_path) 
                    if os.path.isfile(os.path.join(dataset_path, fname)) and fname.lower().endswith(('.jpg', '.jpeg'))]
    elif dataset_name== "kdef":
        face_dir = KDEF_FACE_DIR
        data_paths = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg")):
                    emotion_code = file[4:6]
                    if emotion_code in ['AF', 'AN', 'DI', 'HA', 'SA', 'SU', 'NE']:  # Only select relevant emotions
                        full_path = os.path.join(root, file)
                        data_paths.append(full_path)                                              
    dataset_instance = dataset_class(dataset_name, data_paths, transform=transform, face_dir=face_dir)
    return dataset_instance

def visualize_data(dataset, dataset_name, result_file, num_samples=6):
    if dataset_name == "kmufed":
        class_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    else:
        class_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
        num_samples=7
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))  # Three rows now
    label_selected = set()  # To keep track of selected labels
    selected_indices = []  # Store indices with unique labels

    # Create a list of shuffled indices
    shuffled_indices = list(range(len(dataset)))
    random.shuffle(shuffled_indices)  # Shuffle the dataset indices

    # Loop through the shuffled indices to find one image per label
    for idx in shuffled_indices:
        _, label = dataset[idx]
        if label not in label_selected:
            label_selected.add(label)
            selected_indices.append(idx)
        # Stop once we've selected enough samples
        if len(selected_indices) == num_samples:
            break

    # Visualize the selected samples
    for i, idx in enumerate(selected_indices):
        # Get original, cropped, and masked images
        original_img = dataset.get_original_image(idx)  # Original grayscale image
        cropped_img = np.array(dataset.cropped_faces[idx])  # Cropped face
        masked_img = np.array(dataset.masked_faces[idx])  # Masked face

        _, label = dataset[idx]

        # Plot original image
        axes[0, i].imshow(np.array(original_img), cmap='gray')  # Grayscale for original image
        axes[0, i].set_title(f'Original\nLabel: {class_labels[label]}')
        axes[0, i].axis('off')

        # Plot cropped face
        axes[1, i].imshow(cropped_img, cmap='gray')  # Grayscale for cropped face
        axes[1, i].set_title('Cropped Face')
        axes[1, i].axis('off')

        # Plot masked face
        axes[2, i].imshow(masked_img, cmap='gray')  # Grayscale for masked face
        axes[2, i].set_title('Masked Face')
        axes[2, i].axis('off')

    plt.suptitle(f'{dataset_name} Dataset - Original vs Cropped vs Masked Faces')
    plt.savefig(f"{result_file}/Face_Detection_{dataset_name}.png")
    plt.show()


def create_dataloaders(dataset, dataset_name, batch_size=64, test_size=0.3, val_size=0.5, random_state=42):
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    val_indices, test_indices = train_test_split(temp_indices, test_size=val_size, random_state=random_state)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    dataset.set_mode('train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dataset.set_mode('val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataset.set_mode('test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    file_path = f'{DATALOADERS_PATH}/{dataset_name}_dataloader.pth'
    torch.save({'train_dataset': train_dataset, 'val_dataset': val_dataset, 'test_dataset': test_dataset}, file_path)

    train_count = count_labels(train_dataset, "Training")
    val_count = count_labels(val_dataset, "Validation")
    test_count = count_labels(test_dataset, "Testing")

    dataset_summary = {
        "Dataset": dataset_name,
        "Total images": len(dataset),
        "Training images": len(train_dataset),
        "Validation images": len(val_dataset),
        "Test images": len(test_dataset)
    }

    print(f"\n{dataset_name} Dataset Summary:")
    print(tabulate([dataset_summary], headers="keys", tablefmt="pretty"))

    dataset_summary_df = pd.DataFrame([dataset_summary])
    dataset_summary_file = DATASET_SUMMARY_PATH
    save_dataset_info(dataset_summary_df, train_count, val_count, test_count, dataset_summary_file, dataset_name)
    return train_loader, val_loader, test_loader

def count_labels(dataset, job):
    label_counter = Counter()
    for idx in range(len(dataset)):
        _, label = dataset[idx]  # Only fetch the label, skip the image
        label_counter.update([label])  # Update the counter with the label
    print(f"{job} Count:")
    print(tabulate([label_counter], headers="keys", tablefmt="pretty"))
    label_counter = dict(label_counter)
    labels_df = pd.DataFrame(list(label_counter.items()), columns=["Label", f"{job} Count:"])
    return labels_df

def save_dataset_info(dataset_summary_df, train_labels_df, val_labels_df, test_labels_df, result_file, dataset_name):
    file_path = f"{result_file}/{dataset_name}_summary.csv"
    with open(file_path, mode='w', newline='') as file:
        dataset_summary_df.to_csv(file, index=False, lineterminator='\n')
        file.write("\nNote: Attention masks are included for all samples.\n")
        file.write("Training Label Counts\n")
        train_labels_df.to_csv(file, index=False, lineterminator='\n')
        file.write("Validation Label Counts\n")
        val_labels_df.to_csv(file, index=False, lineterminator='\n')
        file.write("Testing Label Counts\n")
        test_labels_df.to_csv(file, index=False, lineterminator='\n')