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


    def extract_features(self, preprocessed_faces):
        """
        Extracts FaceNet embeddings for preprocessed faces (all faces in one image).
        
        param: preprocessed_faces: List of images with shape (N, H, W, 3), pixel values in [0, 1]
        return: np.ndarray of shape (N, D)
        """
        # Ensure input is always a list and a tensor
        if isinstance(preprocessed_faces, np.ndarray):
            preprocessed_faces = [preprocessed_faces]
        preprocessed_faces = [torch.tensor(face, dtype=torch.float32) for face in preprocessed_faces]

        if not preprocessed_faces:
            return []

        images_tensor = torch.stack(preprocessed_faces).to(self.device)  #(N, H, W, 3)
        images_tensor = images_tensor.permute(0,3,1,2).to(self.device)  # (N, 3, H, W)
        images_tensor = (images_tensor - 0.5) / 0.5  # normalize to [-1,1]

        with torch.no_grad():
            embeddings = self.model(images_tensor) #(N, 512)
        return embeddings.cpu().numpy()

    def batch_extract_features(self, preprocessed_faces_per_image):
        """
        Extracts FaceNet embeddings for a batch of preprocessed faces (all faces in all images).
        param: preprocessed_faces: List of lists of images with shape (N, H, W, 3), pixel values in [0, 1]
        return: List of np.ndarrays of shape (N, D)
        """
        flat_faces = []
        frame_lengths = []
        for i, preprocessed_faces in enumerate(preprocessed_faces_per_image):
            frame_lengths.append(len(preprocessed_faces))
            for face in preprocessed_faces:
                flat_faces.append(face)

        if not flat_faces:
            return []
        embeddings = self.extract_features(flat_faces)
        embeddings_per_image = []
        start = 0
        for length in frame_lengths:
            embeddings_per_image.append(embeddings[start:start+length])
            start += length
        return embeddings_per_image

