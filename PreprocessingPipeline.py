#Credit to https://www.kaggle.com/code/alaasweed/similarity-percentage-using-facenet

import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import numpy as np
import os
import cv2
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

class PreprocessingPipeline:
    def __init__(self, target_size=(160, 160)):
        """
        Initializes the preprocessing pipeline.
        :param target_size: Tuple defining the (width, height) for resizing.
        """
        self.target_size = target_size
        self.detector = MTCNN()  # MTCNN for face detection

    def load_image(self, image_input):
        """
        Loads an image from a file path or uses an already loaded numpy array.
        :param image_input: File path to the image or a numpy array.
        :return: Image in RGB format.
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

    def detect_and_align(self, image):
        """
        Detects the face in the image and returns the cropped face region.
        If no face is detected, raises an error.
        :param image: RGB image.
        :return: Cropped face image.
        """
        # face detection
        results = self.detector.detect_faces(image)
        if results:
            # Use the first detected face (or choose based on confidence/size)
            face = results[0]
            x, y, width, height = face['box']
            # Ensure coordinates are positive
            x, y = abs(x), abs(y)
            # croppin the face from the image
            face_image = image[y:y+height, x:x+width]
            return face_image
        else:
            raise ValueError("No face detected in the image.")


    def detect_and_align_all_faces(self, image):
        """
        Detects and returns all faces in the image. If no faces are found, we
        return the empty list

        param: RGB image
        return: List of faces detected in image
        """
        results = self.detector.detect_faces(image)
        all_faces = []
        if results:
            for face in results:
                x, y, width, height = face['box']
                x, y = abs(x), abs(y)
                face_image = image[y:y+height,x:x+width]
                all_faces.append(face_image)

        return all_faces


    def detect_number_of_faces(self, image):
        """
        Detects and returns the number of faces in the image
        If no face is detected, raises an error.
        :param image: RGB image.
        :return: Number of faces detected
        """

        results = self.detector.detect_faces(image)
        if results:
            return len(results)
        else:
            return 0

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
        Full preprocessing pipeline: loads the image, detects and aligns the face,
        resizes it, and normalizes the pixel values.
        :param image_input: File path or numpy array of the input image.
        :return: Preprocessed image ready for feature extraction.
        """
        # Step 1: Load image
        image = self.load_image(image_input)
        # Step 2: Detect and align face (crop face region)
        face_image = self.detect_and_align(image)
        # Step 3: Resize to target dimensions
        resized_image = self.resize_image(face_image)
        # Step 4: Normalize pixel values
        normalized_image = self.normalize_image(resized_image)
        return normalized_image

    def preprocess_multiple(self, image_input):
        """
        Essentially the same thing as preprocess, except that it preprocesses ALL faces in the 
        image instead of just the highest confidence one.
        """
        image = self.load_image(image_input)
        face_images = self.detect_and_align_all_faces(image)
        resized_images = [self.resize_image(face) for face in face_images]
        normalized_images = [self.normalize_image(image) for image in resized_images]
        return normalized_images
