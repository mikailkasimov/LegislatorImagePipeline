 #Credit to https://www.kaggle.com/code/alaasweed/similarity-percentage-using-facenet

import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import numpy as np
import os
import cv2
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
import PIL._util
if not hasattr(PIL._util, 'is_directory'):
    PIL._util.is_directory = lambda path: os.path.isdir(path)
import torch
# Monkey-patch torch.onnx to add the missing attribute if it doesn't exist.
if not hasattr(torch._C._onnx, "PYTORCH_ONNX_CAFFE2_BUNDLE"):
    torch._C._onnx.PYTORCH_ONNX_CAFFE2_BUNDLE = None

class PreprocessingPipeline:
    def __init__(self, target_size=(160, 160), device='cuda'):
        """
        Initializes the preprocessing pipeline.
        param: --target_size: Tuple defining the (width, height) for resizing.
        """
        self.device = device
        self.target_size = target_size
        self.detector = MTCNN(device=self.device)

    def load_image(self, image_input):
        """
        Loads an image from a file path or uses an already loaded numpy array.
        param: --image_input: File path to the image or a numpy array.
        return: --Image in RGB format.
        """
        if isinstance(image_input, str):
            # read image using OpenCV (BGR format)
            image =     cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Unable to load image from path: {image_input}")
            #convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise TypeError("Input should be a file path or a numpy array.")
        return image   


    def detect_and_crop(self, image):
        """
        Detects and crops, and returns all faces in the image. If no faces are found, we
        return the empty list

        param: RGB image (Height x Width)
        return: -- List: List of RGB image faces
                -- List of Tuples: (x, y, width, height) of box for each face
                -- List of Scalars: Probabilities for each face
        """
        boxes, probs = self.detector.detect(image, landmarks=False)
        faces = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                x1 = max(int(x1),0)
                y1 = max(int(y1),0)
                x2 = min(int(x2), image.shape[1])
                y2 = min(int(y2), image.shape[0])
                face = image[y1:y2, x1:x2]
                faces.append(face)
            return faces, boxes, probs
        else:
            return [], [], []

    def batch_detect_and_crop(self, images):
        """
        Detects and crops, and returns all faces in all images. If no faces are found, we include the empty list for that image

        param: N RGB images (Height x Width)
        return: List of tuples (faces, boxes, probs)
        """
        #Ensure input is always a list
        if isinstance(images, np.ndarray):
            images = [images]

        boxes_batch, probs_batch = self.detector.detect(images, landmarks=False)

        batch_result = []       #gathers faces,boxes,probs  for each image in the batch. list of tuples
        for i, (boxes, probs) in enumerate(zip(boxes_batch, probs_batch)):
            image = images[i]
            faces = []
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    x1 = max(int(x1),0)
                    y1 = max(int(y1),0)
                    x2 = min(int(x2), image.shape[1])
                    y2 = min(int(y2), image.shape[0])
                    face = image[y1:y2, x1:x2]
                    faces.append(face)
                batch_result.append((faces,boxes,probs))
            else:
                batch_result.append(([],[],[]))
        return batch_result     #list of (faces, boxes, probs)

    def resize_image(self, image):
        """
        Resizes the image to the target size.
        :param image: Image array.
        :return: Resized image.
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

    def normalize_image(self, image):
        """
        Normalizes the image pixels to the [0, 1] range.
        :param image: Image array.
        :return: Normalized image array.
        """
        image = image.astype("float32") / 255.0
        return image


    def preprocess(self, image_input):
        """
        Full preprocessing pipeline: loads the image, detects and crops the faces,
        resizes them, and normalizes the pixel values.

        :param image_input: File path or numpy array of the input image.
        :return: Preprocessed image ready for feature extraction.
        """
        image = self.load_image(image_input)
        face_images,_,_ = self.detect_and_crop(image)
        resized_images = [self.resize_image(face) for face in face_images]
        normalized_faces = [self.normalize_image(image) for image in resized_images]
        return normalized_faces

    def batch_preprocess(self, image_inputs):
        """
        Full preprocessing pipeline: loads the images, detects and crops the faces,
        resizes them, and normalizes the pixel values.

        :param image_input: List of file paths or List of numpy arrays of input images
        :return: Preprocessed faces for each image ready for feature extraction.
        """
        images = [self.load_image(image_input) for image_input in image_inputs]
        tuples_per_image = self.batch_detect_and_crop(images)
        normalized_faces_per_image = []
        boxes_per_image = []
        for (faces, boxes, probs) in tuples_per_image:
            resized_images = [self.resize_image(face) for face in faces]
            normalized_faces = [self.normalize_image(face) for face in resized_images]
            normalized_faces_per_image.append(normalized_faces)
            boxes_per_image.append(boxes if boxes is not None else [])
        return normalized_faces_per_image, boxes_per_image