import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
from copy import deepcopy
from tqdm import tqdm
import random
from setup_training_grayscale import device, ensure_dir_exists
import matplotlib.pyplot as plt
from PIL import ImageDraw
# Set the random seed for Python's random module
random.seed(42)
# Set the random seed for numpy
np.random.seed(42)
# Set the random seed for PyTorch
torch.manual_seed(42)
# If using GPU, also set the seed for CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class ImageDataset(Dataset):
    def __init__(self, name, data_paths=None, emotion_map=None, transform=None, face_dir=None):
        self.image_paths = data_paths  # Initialize image_paths
        self.transform = transform
        self.emotion_map = emotion_map
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.name = name
        self.face_dir = face_dir
        self.load_images()

    def load_images(self):
        self.img_list = []
        self.landmarks_list = []
        count = []
        for idx in tqdm(range(len(self.image_paths))):
            img_path = self.image_paths[idx]
            image = Image.open(img_path)  
            
            faces, landmarks = self.detect_faces(image, idx, count)
            self.img_list.append(faces)
            self.landmarks_list.append(landmarks)
        print(f"No face detected in {count}, Total undetected faces: {len(count)}")

    from PIL import ImageDraw

    def detect_faces(self, image, idx, count):
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Detect faces and landmarks using MTCNN
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)

        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)

            # Crop the face using the bounding box
            image = image.crop((x1, y1, x2, y2))

            if landmarks is not None:
                # Take the first set of detected landmarks
                landmarks = landmarks[0]

                # Rescale landmarks to the cropped face
                original_width, original_height = x2 - x1, y2 - y1
                resize_width, resize_height = 224, 224  # Target resized dimensions
                
                # Adjust landmarks to the cropped face
                landmarks_rescaled = []
                for point in landmarks:
                    rescaled_x = (point[0] - x1) * (resize_width / original_width)
                    rescaled_y = (point[1] - y1) * (resize_height / original_height)
                    landmarks_rescaled.append((rescaled_x, rescaled_y))

                landmarks = np.array(landmarks_rescaled)

                # Define bounding boxes for facial regions
                image = image.resize((resize_width, resize_height))
                draw = ImageDraw.Draw(image)

                def draw_region_bounding_box(region_points, color, margin):
                    x_min = min([p[0] for p in region_points])
                    y_min = min([p[1] for p in region_points])
                    x_max = max([p[0] for p in region_points])
                    y_max = max([p[1] for p in region_points])
                    
                    draw.rectangle(
                        [(x_min - margin, y_min - margin), (x_max + margin, y_max + margin)],
                        outline=color,
                        width=2
                    )

                # Define regions based on landmark groups
                eye_points = landmarks[:2]  # Assuming these correspond to eyes
                nose_points = landmarks[2:3]  # Adjusted to include only nose landmarks
                mouth_points = landmarks[3:]  # Adjusted to include all mouth-related landmarks

                # Draw bounding boxes around the regions
                draw_region_bounding_box(eye_points, "blue", margin=35)  # Eye region
                draw_region_bounding_box(nose_points, "green", margin=20)  # Nose region
                draw_region_bounding_box(mouth_points, "red", margin=15)  # Mouth region

            else:
                landmarks = np.zeros((5, 2))
        else:
            count.append(idx)
            landmarks = np.zeros((5, 2))
            
        # Save the RGB version with colorful bounding boxes
        image.save(f"{self.face_dir}/face_with_bbox_regions_{idx}.png")

        # Convert the image to grayscale if needed for further processing
        image = image.convert('L')
        return image, landmarks



    def get_original_image(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        return image
    
    def __len__(self):
        return len(self.image_paths)

    def extract_label(self, filename):
        raise NotImplementedError("This method should be overridden by subclasses")

    def __getitem__(self, idx):
        image = deepcopy(self.img_list[idx])
        landmarks = deepcopy(self.landmarks_list[idx])
        image, landmarks = self.transform(image, landmarks)
        img_path = self.image_paths[idx]
        label = self.extract_label(os.path.basename(img_path))
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        return image, label, landmarks
    
class KMUFEDDataset(ImageDataset):
    def __init__(self, name, image_paths, transform=None, face_dir=None):
        emotion_map = {'AN': 0, 'DI': 1, 'FE': 2, 'HA': 3, 'SA': 4, 'SU': 5}
        super().__init__(name, image_paths, emotion_map, transform, face_dir)
        
    def extract_label(self, filename):
        emotion_code = filename.split('_')[1]
        return self.emotion_map[emotion_code]

class KDEFDataset(ImageDataset):
    def __init__(self, name, image_paths, transform=None,face_dir=None):
        emotion_map = {'AN': 0, 'DI': 1, 'AF': 2, 'HA': 3, 'SA': 4, 'SU': 5} # AF = fear, AN = angry
        super().__init__(name, image_paths, emotion_map, transform, face_dir)
        
    def extract_label(self, filename):
        emotion_code = filename[4:6]
        return self.emotion_map.get(emotion_code, -1)


class CsvDataset(Dataset):
    def __init__(self, name, data_paths, emotion_map=None, transform=None, face_dir=None):
        self.data = pd.read_csv(data_paths)
        self.emotion_map = emotion_map
        self.data = self.data[self.data['emotion'].isin(self.emotion_map.keys())]
        self.transform = transform
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.name = name
        self.face_dir = face_dir
        self.load_images()

    def load_images(self):
        self.img_list = []
        self.landmarks_list = []
        count = []
        for idx in tqdm(range(len(self.data))):
            row = self.data.iloc[idx]
            image = np.fromstring(row['pixels'], dtype=int, sep=' ').reshape(48, 48)
    
            # blurred_image_np = cv2.GaussianBlur(image.astype(np.uint8), (1, 1), 0) # Apply Gaussian blur
            image = Image.fromarray(image)
            
            faces, landmarks = self.detect_faces(image, idx, count) # Detect faces and store the processed image
            self.img_list.append(faces)
            self.landmarks_list.append(landmarks)
        print(f"No face detected in {count}, Total undetected faces: {len(count)}")
    
    def detect_faces(self, image, idx, count):
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Detect faces and landmarks using MTCNN
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)

        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)

            # Crop the face using the bounding box
            image = image.crop((x1, y1, x2, y2))

            if landmarks is not None:
                # Take the first set of detected landmarks
                landmarks = landmarks[0]

                # Rescale landmarks to the cropped face
                original_width, original_height = x2 - x1, y2 - y1
                resize_width, resize_height = 224, 224  # Target resized dimensions
                
                # Adjust landmarks to the cropped face
                landmarks_rescaled = []
                for point in landmarks:
                    rescaled_x = (point[0] - x1) * (resize_width / original_width)
                    rescaled_y = (point[1] - y1) * (resize_height / original_height)
                    landmarks_rescaled.append((rescaled_x, rescaled_y))

                landmarks = np.array(landmarks_rescaled)

                # Define bounding boxes for facial regions
                image = image.resize((resize_width, resize_height))
                draw = ImageDraw.Draw(image)

                def draw_region_bounding_box(region_points, color, margin):
                    x_min = min([p[0] for p in region_points])
                    y_min = min([p[1] for p in region_points])
                    x_max = max([p[0] for p in region_points])
                    y_max = max([p[1] for p in region_points])
                    
                    draw.rectangle(
                        [(x_min - margin, y_min - margin), (x_max + margin, y_max + margin)],
                        outline=color,
                        width=2
                    )

                # Define regions based on landmark groups
                eye_points = landmarks[:2]  # Assuming these correspond to eyes
                nose_points = landmarks[2:3]  # Adjusted to include only nose landmarks
                mouth_points = landmarks[3:]  # Adjusted to include all mouth-related landmarks

                # Draw bounding boxes around the regions
                draw_region_bounding_box(eye_points, "blue", margin=35)  # Eye region
                draw_region_bounding_box(nose_points, "green", margin=20)  # Nose region
                draw_region_bounding_box(mouth_points, "red", margin=15)  # Mouth region

            else:
                landmarks = np.zeros((5, 2))
        else:
            count.append(idx)
            landmarks = np.zeros((5, 2))
            
        # Save the RGB version with colorful bounding boxes
        image.save(f"{self.face_dir}/face_with_bbox_regions_{idx}.png")

        # Convert the image to grayscale if needed for further processing
        image = image.convert('L')
        return image, landmarks

    def get_original_image(self, idx):
        row = self.data.iloc[idx]
        image_data = np.fromstring(row['pixels'], dtype=int, sep=' ').reshape(48, 48)
        image = Image.fromarray(image_data)
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = deepcopy(self.img_list[idx])
        landmarks = deepcopy(self.landmarks_list[idx])
        image, landmarks = self.transform(image, landmarks)

        label = self.emotion_map[self.data.iloc[idx]['emotion']]
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        return image, label, landmarks


class FER2013Dataset(CsvDataset):
    def __init__(self, name, csv_file, transform=None, face_dir=None):
        emotion_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}  # FER2013 emotion mapping
        super().__init__(name, csv_file, emotion_map, transform, face_dir)


# class CKPlusDataset(CsvDataset):
#     def __init__(self, csv_file, transform=None):
#         emotion_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}  # CKPlus emotion mapping
#         super().__init__(csv_file, emotion_map, transform)


# class AIDEDataset(BaseDataset):
#     def __init__(self, image_paths, transform=None):
#         super().__init__(image_paths, transform)
#         self.emotion_map = {
#             'Anger': 0, 'Anxiety': 1, 'Peace': 2, 'Weariness': 3, 'Happiness': 4
#         }
#         self.load_images()  # Load images directly without face detection

#     def load_images(self):
#         self.img_list = []
#         for idx in tqdm(range(len(self.image_paths))):
#             img_path = self.image_paths[idx]
#             image = Image.open(img_path).convert("L")  # Convert to grayscale
            
#             # Directly append the image without face detection
#             self.img_list.append(image)

#     def extract_label(self, filename):
#         emotion_label = filename.split('_')[0]
#         label = self.emotion_map.get(emotion_label, -1)
#         if label == -1:
#             print(f"Warning: Label for {filename} not found in emotion_map.")
#         return label