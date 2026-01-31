from abc import ABC, abstractmethod
from typing import Any
import torch
from facenet_pytorch import InceptionResnetV1
from FeatureExtractor import FeatureExtractor
import numpy as np

class FacenetFeatureExtractor(FeatureExtractor):
    def __init__(self,device='cuda'):
        self.device: str = device
        self.model: Any = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        

    def extract_features(self, faces):
            # 1. Handle empty input
            if faces is None or len(faces) == 0:
                return np.empty((0, 512), dtype=np.float32)

            # 2. Convert to tensor and ensure 4D (N, H, W, 3)
            if isinstance(faces, np.ndarray):
                images_tensor = torch.from_numpy(faces).float()
            elif isinstance(faces, torch.Tensor):
                images_tensor = faces.float()
            else:
                # Handle list of images by stacking them
                images_tensor = torch.stack([torch.from_numpy(np.asarray(f)).float() for f in faces])

            # If a single image (H, W, 3) was passed, make it (1, H, W, 3)
            if images_tensor.ndimension() == 3:
                images_tensor = images_tensor.unsqueeze(0)

            # 3. Permute to (N, 3, H, W) - This now always works because input is forced to 4D
            images_tensor = images_tensor.permute(0, 3, 1, 2).to(self.device)
            
            # 4. Normalize and Forward Pass
            images_tensor = (images_tensor - 0.5) / 0.5
            with torch.no_grad():
                embeddings = self.model(images_tensor)
            return embeddings.cpu().numpy()

    def batch_extract_features(self, faces_list):
        flat_faces = []
        frame_lengths = []
        
        for preprocessed_faces in faces_list:
            # Check if current element is empty
            if preprocessed_faces is None or len(preprocessed_faces) == 0:
                frame_lengths.append(0)
                continue
            
            # Record length (N) and add faces to the flat list
            # Since preprocessed_faces is (N, H, W, 3), we iterate through N
            num_faces = len(preprocessed_faces)
            frame_lengths.append(num_faces)
            for face in preprocessed_faces:
                flat_faces.append(face)

        if not flat_faces:
            # Return a list of empty arrays matching input length for consistency
            return [np.empty((0, 512), dtype=np.float32) for _ in faces_list]

        # Use the updated extract_features which handles the list/batch properly
        all_embeddings = self.extract_features(flat_faces)

        # Re-group embeddings back to match the original structure
        embeddings_per_image = []
        start = 0
        for length in frame_lengths:
            if length == 0:
                embeddings_per_image.append(np.empty((0, 512), dtype=np.float32))
            else:
                embeddings_per_image.append(all_embeddings[start : start + length])
                start += length
        return embeddings_per_image