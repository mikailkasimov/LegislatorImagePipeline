from abc import ABC, abstractmethod
from typing import Any, Sequence, overload, Tuple
from numpy.typing import NDArray
import numpy as np
import cv2

class FaceExtractor(ABC):
    def __init__(self):
        self.device: str = None
        self.model: Any = None
        self.target_size: Tuple[int,int] = None


    @abstractmethod
    def preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """ 
        Preprocess given face (i.e normalize, resize, etc...) and return the preprocessed image
        """
        ...

    @abstractmethod
    def batch_preprocess(self, input_list: Sequence[NDArray[np.uint8]]) -> Sequence[NDArray[np.uint8]]:
        """ 
        Preprocess given faces (i.e normalize, resize, etc...) and return all preprocessed images

        param: input_list: 
        """
        ...

    @abstractmethod
    def extract_faces(self, input: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """ 
        Extract faces from image sample

        """
        ...

    @abstractmethod
    def batch_extract_faces(self, input_list: Sequence[NDArray[np.uint8]]) -> Sequence[NDArray[np.uint8]]:
        """ 
        Extract faces from all image samples

        """
        ...

    def load_image(self, image_input) -> NDArray[np.uint8]:
        if isinstance(image_input, str):
            try:
                image = cv2.imread(image_input, cv2.IMREAD_COLOR) #bgr, 3 channels
            except Exception as e:
                print(f"Error: {e}")
        
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise TypeError("Input must be filename or np.ndarray")
        return image

    def resize_image(self, image: NDArray[np.uint8], target_size: Tuple[int, int]) -> NDArray[np.uint8]:
        return cv2.resize(image,target_size,interpolation=cv2.INTER_AREA)

    def normalize_image(self, image):
        return image.astype("float32") / 255.0

        
    def _to_rgb_image(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Ensure the input is an RGB image shaped (H, W, 3).
        Raises a clear error for non-image inputs (e.g., embeddings).

        param: input: (H,W,3) np.ndarray input image
        """
        arr = np.asarray(image)
        if arr.ndim == 2:
            raise ValueError(
                "Received a 2D array. Face detection expects an RGB image shaped "
                "(H, W, 3); if you have embeddings or another 2D tensor, pass the "
                "original image frames instead."
            )
        if arr.ndim == 3:
            if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 4:
                arr = arr[..., :3]
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.shape[2] == 3:
                return arr
        raise ValueError(f"Expected image shaped (H, W, 3). Got shape {arr.shape}.")
