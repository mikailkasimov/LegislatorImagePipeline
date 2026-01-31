from facenet_pytorch import MTCNN
from typing import Any, Sequence, overload, Tuple
from numpy.typing import NDArray
import numpy as np
from FaceExtractor import FaceExtractor


class MTCNNFaceExtractor(FaceExtractor):
    def __init__(self, device='cuda', **kwargs):
        self.device = device
        self.model = MTCNN(device=device,**kwargs)
        self.target_size = (160,160) #MTCNN was trained on 160,160,3

    def preprocess(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """ 
        Preprocess given face (i.e normalize, resize, etc...) for embedding extraction

        return: Tuple of ( (H,W,3) preprocessed image, 
                           (H,W) box, 
                           Scalar prob
                         )
        """
        image = self._to_rgb_image(self.load_image(image))
        faces, boxes, probs = self.extract_faces(image)
        resized_faces = [self.resize_image(face, self.target_size) for face in faces]
        normalized_faces = [self.normalize_image(face) for face in resized_faces]
        return np.asarray(normalized_faces), np.asarray(boxes), np.asarray(probs)
        
    def batch_preprocess(self, image_list: Sequence[NDArray[np.uint8]]) -> Sequence[NDArray[np.uint8]]:
        """ 
        Preprocess given faces (i.e normalize, resize, etc...) for embedding extraction

        param: input_list: Sequence[(N,H,W,3)] of (H,W,3) np.ndarray images
             | input: list(str), list of image paths

        return: Tuple of ( (N,H,W,3) list of preprocessed images, 
                           (N,H,W) list of boxes, 
                           (N) list of probs 
                         )
        """
        images = [self.load_image(x) for x in image_list]
        out = []
        for img in images:
            img = self._to_rgb_image(img)
            faces, boxes, probs = self.extract_faces(img)  # faces: list[RGB uint8]
            target = self.target_size
            resized_faces = [self.resize_image(face, target) for face in faces]
            normalized_faces = [self.normalize_image(f) for f in resized_faces]

            if len(normalized_faces) > 0:
                normalized_faces = np.asarray(normalized_faces, dtype=np.float32)
            else:
                normalized_faces = np.empty((0, target[1], target[0], 3), dtype=np.float32)
            boxes = boxes.astype(np.float32) if isinstance(boxes, np.ndarray) else np.empty((0, 4), dtype=np.float32)
            probs = probs.astype(np.float32) if isinstance(probs, np.ndarray) else np.empty((0,), dtype=np.float32)
            out.append((normalized_faces, boxes, probs))
        return out
        

    def extract_faces(self, image: NDArray[np.uint8], **kwargs) -> Tuple[list[np.ndarray], NDArray[np.float32], NDArray[np.float32]]:
        """ 
        Extract faces from image sample, and returns metadata (including cropped face)

        param: image: (H,W,3) np.ndarray input image
        return: Tuple[(?,H,W,3), (?,4), (?,)] of np.ndarray representing face(s),box(s),prob(s) for each face (?)
        """ 
        #print(image.shape)
        boxes, probs = self.model.detect(
            image, 
            landmarks=False, 
            **kwargs
        )
        faces: list[np.ndarray] = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                x1 = max(int(x1),0)
                y1 = max(int(y1),0)
                x2 = min(int(x2), image.shape[1])
                y2 = min(int(y2), image.shape[0])
                face = image[y1:y2, x1:x2]
                faces.append(face)
            boxes_arr = np.asarray(boxes, dtype=np.float32)
            probs_arr = np.asarray(probs, dtype=np.float32) if probs is not None else np.empty((0,), dtype=np.float32)
            return faces, boxes_arr, probs_arr
        return ([], np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32))
        

    def batch_extract_faces(self, image_list: Sequence[NDArray[np.uint8]]) -> Sequence[NDArray[np.uint8]]:
        """ 
        Extract faces from all image samples

        param: image_list: (N, H, W, 3) np.ndarray image input tensor
        return: Tuple[(N,?,H,W,3), (N,?,4), (N,?,)] representing face(s),box(s),prob(s) for each face (?) for N images
        """
        boxes_batch, probs_batch = self.model.detect(image_list, landmarks=False)

        batch_result = []       #gathers faces,boxes,probs  for each image in the batch. list of tuples
        for i, (boxes, probs) in enumerate(zip(boxes_batch, probs_batch)):
            image = image_list[i]
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
                batch_result.append((np.ndarray([]),np.ndarray([]),np.ndarray([])))
        return batch_result 
