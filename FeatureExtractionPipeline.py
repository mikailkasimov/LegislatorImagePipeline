#Credit to https://www.kaggle.com/code/alaasweed/similarity-percentage-using-facenet

import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import cv2
import numpy as np
import os
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from mtcnn import MTCNN
import PIL._util
if not hasattr(PIL._util, 'is_directory'):
    PIL._util.is_directory = lambda path: os.path.isdir(path)
import torch
# Monkey-patch torch.onnx to add the missing attribute if it doesn't exist.
if not hasattr(torch._C._onnx, "PYTORCH_ONNX_CAFFE2_BUNDLE"):
    torch._C._onnx.PYTORCH_ONNX_CAFFE2_BUNDLE = None

class FeatureExtractionPipeline:
    def __init__(self, device='cuda'):
        """
        Initializes the FaceNet (InceptionResnetV1) model for feature extraction.
        Loads pretrained weights (e.g., trained on VGGFace2) and sets the model to eval mode.
        """
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Def a transform that first converts float32 images (range [0,1]) to uint8,
        # then converts to PIL, resizes to 160x160, converts to tensor, and normalizes to [-1,1].
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: (x * 255).astype(np.uint8) if x.dtype == np.float32 else x),
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    
    def extract_features(self, preprocessed_image):
        """
        Extracts features from a preprocessed image using FaceNet.
        
        Parameters:
            preprocessed_image (numpy.ndarray): Preprocessed image (e.g., shape (224,224,3), values in [0,1])
        
        Returns:
            numpy.ndarray: A flattened feature vector (typically 512-dimensional).
        """
        # Apply the transformation to get a tensor of shape (3, 160, 160)
        img_tensor = self.transform(preprocessed_image)
        # Add a batch dimension so the tensor becomes (1, 3, 160, 160)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Forward pass through the FaceNet model to obtain the embedding.
        with torch.no_grad():
            features = self.model(img_tensor)
        
        # Squeeze the output to remove the batch dimension and convert to a NumPy array.
        features = features.squeeze().cpu().numpy()
        return features

    def batch_extract_features(self, preprocessed_images):
        """
        Extracts FaceNet embeddings for a batch of preprocessed images.
        
        param: preprocessed_images: List of images (or a single image) with shape (H, W, 3), pixel values in [0, 1]
        return: np.ndarray of shape (N, D)
        """
        if isinstance(preprocessed_images, np.ndarray):
            preprocessed_images = [preprocessed_images]

        tensors = []
        for img in preprocessed_images:
            tensor = self.transform(img)  # shape: (3, 160, 160)
            tensors.append(tensor)

        batch = torch.stack(tensors).to(self.device)  # shape: (N, 3, 160, 160)

        with torch.no_grad():
            embeddings = self.model(batch)  # shape: (N, 512)

        return embeddings.cpu().numpy()  # shape: (N, 512)
