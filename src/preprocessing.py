from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Augmentation:
    image: np.ndarray

    def rotation(self, angle: int) -> None:
        raise NotImplementedError

    def horizontal_flip(self, angle: int) -> None:
        raise NotImplementedError

    def vertical_flip(self, angle: int) -> None:
        raise NotImplementedError

    def zoom(self, angle: int) -> None:
        raise NotImplementedError

    def width_shift(self, angle: int) -> None:
        raise NotImplementedError

    def height_shift(self, angle: int) -> None:
        raise NotImplementedError
    
    def sheer(self, angle: int) -> None:
        raise NotImplementedError

    def brightness(self, angle: int) -> None:
        raise NotImplementedError

    def constrast(self, angle: int) -> None:
        raise NotImplementedError


@dataclass
class Preprocessing:
    image: np.ndarray

    def resize(self, width: int, height: int) -> None: 
        raise NotImplementedError

    def grayscale(self) -> None: 
        raise NotImplementedError

    def gaussian_blur(self) -> None: 
        raise NotImplementedError

    def edge_detection(self) -> None: 
        raise NotImplementedError

    def normalization(self) -> None: 
        raise NotImplementedError


