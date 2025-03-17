import os
import torch
import numpy as np
import pandas as pd
import random
from PIL import ImageDraw, Image
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
from copy import deepcopy
from tqdm import tqdm
from setup_training_grayscale import device, transform
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
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
        self.cropped_faces = []  # Cropped face images
        self.masked_faces = []
        self.mode = 'train'
        self.load_images()

    def set_mode(self, mode):
        """Set the mode for the dataset: 'train', 'val', or 'test'."""
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val', or 'test'."
        self.mode = mode

    def load_images(self):
        
        count = []
        for idx in tqdm(range(len(self.image_paths))):
            img_path = self.image_paths[idx]
            image = Image.open(img_path)  
            
            # Detect faces and generate the attention mask
            cropped_face, masked_face = self.detect_faces(image, idx, count) ## Change
            
            self.cropped_faces.append(cropped_face)
            self.masked_faces.append(masked_face)
            
        print(f"No face detected in {count}, Total undetected faces: {len(count)}")

    # def detect_faces(self, image, idx, count):
    #     if image.mode != 'RGB':
    #         image = image.convert('RGB')

    #     # Detect faces and landmarks using MTCNN
    #     boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
    #     image = image.convert('L')
    #     if boxes is not None and len(boxes) > 0:
    #         x1, y1, x2, y2 = [int(b) for b in boxes[0]]
    #         x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)

    #         # Crop the face using the bounding box
    #         cropped_face = image.crop((x1, y1, x2, y2)).resize((224, 224))

    #         if landmarks is not None:
    #             # Process landmarks and generate attention mask
    #             landmarks = landmarks[0]
    #             original_width, original_height = x2 - x1, y2 - y1
    #             resize_width, resize_height = 224, 224  # Target resized dimensions

    #             # Rescale landmarks to the cropped face
    #             landmarks_rescaled = []
    #             for point in landmarks:
    #                 rescaled_x = (point[0] - x1) * (resize_width / original_width)
    #                 rescaled_y = (point[1] - y1) * (resize_height / original_height)
    #                 landmarks_rescaled.append((rescaled_x, rescaled_y))
    #             landmarks = np.array(landmarks_rescaled)

    #             # Create the attention mask
    #             attention_mask = Image.new("L", (resize_width, resize_height), 128)
    #             mask_draw = ImageDraw.Draw(attention_mask)

    #             def draw_attention_mask(region_points, margin, intensity):
    #                 x_min = min([p[0] for p in region_points])
    #                 y_min = min([p[1] for p in region_points])
    #                 x_max = max([p[0] for p in region_points])
    #                 y_max = max([p[1] for p in region_points])

    #                 # Add margin to create a broader mask
    #                 mask_draw.rectangle(
    #                     [(x_min - margin, y_min - margin), (x_max + margin, y_max + margin)],
    #                     fill=intensity
    #                 )

    #             # Define regions and draw masks
    #             eye_points = landmarks[:2]
    #             nose_points = landmarks[2:3]
    #             mouth_points = landmarks[3:]
    #             draw_attention_mask(eye_points, margin=35, intensity=255)
    #             draw_attention_mask(nose_points, margin=20, intensity=255)
    #             draw_attention_mask(mouth_points, margin=20, intensity=255)

    #             # Resize and apply mask to the cropped face
    #             attention_mask = attention_mask.resize((resize_width, resize_height))
    #             cropped_face_array = np.array(cropped_face)
    #             mask_array = np.array(attention_mask) / 255.0
    #             enhanced_face = cropped_face_array * mask_array + cropped_face_array * 0.5 * (1 - mask_array)
    #             enhanced_face = enhanced_face.astype(np.uint8)
    #             masked_face = Image.fromarray(enhanced_face)
    #         else:
    #             masked_face = cropped_face
    #     else:
    #         count.append(idx)
    #         cropped_face = image.resize((224, 224))
    #         masked_face = cropped_face

    #     return cropped_face, masked_face
    def detect_faces(self, image, idx, count):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Detect faces and landmarks using MTCNN
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        image = image.convert('L')  # Convert to grayscale
        
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)

            # Crop the face and resize
            cropped_face = image.crop((x1, y1, x2, y2)).resize((224, 224))
            
            if landmarks is not None:
                # Rescale landmarks to match resized face
                landmarks = landmarks[0]
                original_width, original_height = x2 - x1, y2 - y1
                resize_width, resize_height = 224, 224
                
                rescaled_landmarks = [
                    ((point[0] - x1) * (resize_width / original_width),
                    (point[1] - y1) * (resize_height / original_height))
                    for point in landmarks
                ]
                
                # Create an empty mask
                attention_mask = Image.new("L", (resize_width, resize_height), 0)
                mask_draw = ImageDraw.Draw(attention_mask)
                
                def draw_smooth_mask(center_x, center_y, radius, intensity):
                    bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
                    mask_draw.ellipse(bbox, fill=intensity)
                
                # Define feature regions and draw ellipses with smooth transitions
                eye_radius = 35
                nose_radius = 25

                # Mouth: Use both corners for proper coverage
                mouth_points = rescaled_landmarks[3:5]  # Two corners of the mouth
                x_min = min(mouth_points[0][0], mouth_points[1][0])
                y_min = min(mouth_points[0][1], mouth_points[1][1])
                x_max = max(mouth_points[0][0], mouth_points[1][0])
                y_max = max(mouth_points[0][1], mouth_points[1][1])
                mouth_center_x = (x_min + x_max) / 2
                mouth_center_y = (y_min + y_max) / 2
                mouth_radius = (x_max - x_min) // 2 + 10  # Expand the mouth mask

                draw_smooth_mask(*rescaled_landmarks[0], eye_radius, 255)  # Left eye
                draw_smooth_mask(*rescaled_landmarks[1], eye_radius, 255)  # Right eye
                draw_smooth_mask(*rescaled_landmarks[2], nose_radius, 255)  # Nose
                draw_smooth_mask(mouth_center_x, mouth_center_y, mouth_radius, 255)  # Full mouth

                # Apply Gaussian blur to smooth transitions
                attention_mask = attention_mask.filter(ImageFilter.GaussianBlur(10))
                
                # Convert cropped face and mask to numpy arrays
                cropped_face_array = np.array(cropped_face, dtype=np.float32)
                mask_array = np.array(attention_mask, dtype=np.float32) / 255.0
                
                # Blend the mask with the image smoothly
                enhanced_face = cropped_face_array * mask_array + cropped_face_array * 0.5 * (1 - mask_array)
                enhanced_face = enhanced_face.astype(np.uint8)
                masked_face = Image.fromarray(enhanced_face)
            else:
                masked_face = cropped_face
        else:
            count.append(idx)
            cropped_face = image.resize((224, 224))
            masked_face = cropped_face
        
        return cropped_face, masked_face

    
    def get_original_image(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        return image
    
    def __len__(self):
        return len(self.image_paths)

    def extract_label(self, filename):
        raise NotImplementedError("This method should be overridden by subclasses")

    def __getitem__(self, idx):
        if self.mode == 'train': 
            image = deepcopy(self.masked_faces[idx])  # Use masked images for training
        else:
            image = deepcopy(self.cropped_faces[idx])

        image = self.transform(image)
        
        img_path = self.image_paths[idx]
        label = self.extract_label(os.path.basename(img_path))
        
        return image, label
    
class KMUFEDDataset(ImageDataset):
    def __init__(self, name, image_paths, transform=None, face_dir=None):
        emotion_map = {'AN': 0, 'DI': 1, 'FE': 2, 'HA': 3, 'SA': 4, 'SU': 5}
        super().__init__(name, image_paths, emotion_map, transform, face_dir)
        
    def extract_label(self, filename):
        emotion_code = filename.split('_')[1]
        return self.emotion_map[emotion_code]

class KDEFDataset(ImageDataset):
    def __init__(self, name, image_paths, transform=None,face_dir=None):
        emotion_map = {'AN': 0, 'DI': 1, 'AF': 2, 'HA': 3, 'SA': 4, 'SU': 5, 'NE':6} # AF = fear, AN = angry
        super().__init__(name, image_paths, emotion_map, transform, face_dir)
        
    def extract_label(self, filename):
        emotion_code = filename[4:6]
        return self.emotion_map.get(emotion_code, -1)


class CsvDataset(Dataset):
    def __init__(self, name, data_paths, emotion_map=None, transform=None, face_dir=None):
        self.data = pd.read_csv(data_paths)
        self.transform = transform
        self.emotion_map = emotion_map
        self.device = device
        self.data = self.data[self.data['emotion'].isin(self.emotion_map.keys())]
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.name = name
        self.face_dir = face_dir
        self.cropped_faces = []  # Cropped face images
        self.masked_faces = []
        self.mode = 'train'
        self.load_images()

    def set_mode(self, mode):
        """Set the mode for the dataset: 'train', 'val', or 'test'."""
        assert mode in ['train', 'val', 'test'], "Mode must be 'train', 'val', or 'test'."
        self.mode = mode

    def load_images(self):
        
        count = []
        for idx in tqdm(range(len(self.data))):
            row = self.data.iloc[idx]
            image = np.fromstring(row['pixels'], dtype=int, sep=' ').reshape(48, 48)
            image = Image.fromarray(image)
            cropped_face, masked_face = self.detect_faces(image, idx, count) # Detect faces and store the processed image
            self.cropped_faces.append(cropped_face)
            self.masked_faces.append(masked_face)
        print(f"No face detected in {count}, Total undetected faces: {len(count)}")
    
    def detect_faces(self, image, idx, count):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Detect faces and landmarks using MTCNN
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        image = image.convert('L')  # Convert to grayscale
        
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)

            # Crop the face and resize
            cropped_face = image.crop((x1, y1, x2, y2)).resize((224, 224))
            
            if landmarks is not None:
                # Rescale landmarks to match resized face
                landmarks = landmarks[0]
                original_width, original_height = x2 - x1, y2 - y1
                resize_width, resize_height = 224, 224
                
                rescaled_landmarks = [
                    ((point[0] - x1) * (resize_width / original_width),
                    (point[1] - y1) * (resize_height / original_height))
                    for point in landmarks
                ]
                
                # Create an empty mask
                attention_mask = Image.new("L", (resize_width, resize_height), 0)
                mask_draw = ImageDraw.Draw(attention_mask)
                
                def draw_smooth_mask(center_x, center_y, radius, intensity):
                    bbox = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
                    mask_draw.ellipse(bbox, fill=intensity)
                
                # Define feature regions and draw ellipses with smooth transitions
                eye_radius = 35
                nose_radius = 25

                # Mouth: Use both corners for proper coverage
                mouth_points = rescaled_landmarks[3:5]  # Two corners of the mouth
                x_min = min(mouth_points[0][0], mouth_points[1][0])
                y_min = min(mouth_points[0][1], mouth_points[1][1])
                x_max = max(mouth_points[0][0], mouth_points[1][0])
                y_max = max(mouth_points[0][1], mouth_points[1][1])
                mouth_center_x = (x_min + x_max) / 2
                mouth_center_y = (y_min + y_max) / 2
                mouth_radius = (x_max - x_min) // 2 + 10  # Expand the mouth mask

                draw_smooth_mask(*rescaled_landmarks[0], eye_radius, 255)  # Left eye
                draw_smooth_mask(*rescaled_landmarks[1], eye_radius, 255)  # Right eye
                draw_smooth_mask(*rescaled_landmarks[2], nose_radius, 255)  # Nose
                draw_smooth_mask(mouth_center_x, mouth_center_y, mouth_radius, 255)  # Full mouth

                # Apply Gaussian blur to smooth transitions
                attention_mask = attention_mask.filter(ImageFilter.GaussianBlur(10))
                
                # Convert cropped face and mask to numpy arrays
                cropped_face_array = np.array(cropped_face, dtype=np.float32)
                mask_array = np.array(attention_mask, dtype=np.float32) / 255.0
                
                # Blend the mask with the image smoothly
                enhanced_face = cropped_face_array * mask_array + cropped_face_array * 0.5 * (1 - mask_array)
                enhanced_face = enhanced_face.astype(np.uint8)
                masked_face = Image.fromarray(enhanced_face)
            else:
                masked_face = cropped_face
        else:
            count.append(idx)
            cropped_face = image.resize((224, 224))
            masked_face = cropped_face
        
        return cropped_face, masked_face

    def get_original_image(self, idx):
        row = self.data.iloc[idx]
        image_data = np.fromstring(row['pixels'], dtype=int, sep=' ').reshape(48, 48)
        image = Image.fromarray(image_data)
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'train': 
            image = deepcopy(self.masked_faces[idx])  # Use masked images for training
        else:
            image = deepcopy(self.cropped_faces[idx])
        image = self.transform(image)
        label = self.emotion_map[self.data.iloc[idx]['emotion']]
        return image, label

class FER2013Dataset(CsvDataset):
    def __init__(self, name, csv_file, transform=None, face_dir=None):
        emotion_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}  # FER2013 emotion mapping
        super().__init__(name, csv_file, emotion_map, transform, face_dir)