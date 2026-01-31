from abc import ABC, abstractmethod
from typing import Any

class FeatureExtractor(ABC):
    def __init__(self,device='cuda'):
        self.device: str = None
        self.model: Any = None

    @abstractmethod
    def extract_features(self, input: Any) -> Any:
        ...

    @abstractmethod
    def batch_extract_features(self: Any) -> Any:
        ...