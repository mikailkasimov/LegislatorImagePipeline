import numpy as np
from typing import Tuple
from numpy.typing import NDArray
import cv2

class VideoHandler:
    def __init__(self, device="cuda"):
        self.device=device

    def get_frames(self, video: str, interval: int) -> Tuple[NDArray[np.floating], NDArray[np.uint8]]:
        """
        Sample frames every frame_rate amount of seconds

        param: video: str, mp4 file path of the video
        param: interval: str, sample 1 frame every interval seconds

        return: list of frames
        """
        frames = []
        frames_index = []
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)  # Number of frames to skipjk
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            if frame_count % frame_interval == 0:
                frames.append(frame)
                frames_index.append(frame_count)
            frame_count += 1
        cap.release()
        return np.array(frames), np.array(frames_index)

    def get_frame_generator(self, video, interval, batch_size):
        """
        Sample frames every frame_rate amount of seconds and yields every batch_size

        param: frame_rate, sample 1 frame every frame_rate seconds
        param: video, mp4 file path of the video

        return: list of frames
        """
        frames = []
        frames_index = []
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame_count % frame_interval) == 0:
                frames.append(frame)
                frames_index.append(frame_count)
            frame_count += 1
            if len(frames) == batch_size:
                yield np.array(frames), np.array(frames_index)
                frames = []
                frames_index = []
        #yield leftover frames that did not full batch
        if frames:
            yield np.array(frames), np.array(frames_index)
        cap.release()
