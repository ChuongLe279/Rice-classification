from dataclasses import dataclass
import cv2
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Augmentation:
    image: np.ndarray

    def rotate_image(self, angle: int) -> np.ndarray:
        image_center = tuple(np.array(self.image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(self.image, rot_mat, self.image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def horizontal_flip(self) -> np.ndarray:
        return cv2.flip(self.image, 1)

    def vertical_flip(self) -> np.ndarray:
        return cv2.flip(self.image, 0)

    def zoom_center(self, zoom_factor: float) -> np.ndarray:
        if zoom_factor <= 0:
            raise ValueError("zoom_factor must be greater than 0.")

        h, w = self.image.shape[:2]
        center = ((w - 1) / 2, (h - 1) / 2)
        trans_mat = cv2.getRotationMatrix2D(center, 0, zoom_factor)

        return cv2.warpAffine(self.image, trans_mat, (w, h))

    def width_shift(self, offset_x: int) -> np.ndarray:
        h, w = self.image.shape[:2]
        trans_mat = np.array(
            [[1, 0, offset_x], [0, 1, 0]],
            dtype=np.float32,
        )
        return cv2.warpAffine(self.image, trans_mat, (w, h))

    def height_shift(self, offset_y: int) -> np.ndarray:
        h, w = self.image.shape[:2]
        trans_mat = np.array(
            [[1, 0, 0], [0, 1, offset_y]],
            dtype=np.float32,
        )
        return cv2.warpAffine(self.image, trans_mat, (w, h))
    
    def shear(self, shear_factor: float = 0.2) -> np.ndarray:
        h, w = self.image.shape[:2]
        trans_mat = np.array(
            [[1, shear_factor, -shear_factor * (h - 1) / 2],
            [0, 1, 0]],
            dtype=np.float32,
        )
        return cv2.warpAffine(self.image, trans_mat, (w, h))

    def brightness(self, brightness: int) -> np.ndarray:
        adjusted = self.image.astype(np.float32) + brightness
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def contrast(self, contrast: float) -> np.ndarray:
        if contrast < 0:
            raise ValueError("contrast must be non-negative.")

        adjusted = (self.image.astype(np.float32) - 127.5) * contrast + 127.5
        return np.clip(adjusted, 0, 255).astype(np.uint8)


@dataclass
class Preprocessing:
    image: np.ndarray

    def resize(self, width: int, height: int) -> np.ndarray: 
        return cv2.resize(self.image, (width, height))

    def grayscale(self) -> np.ndarray: 
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self) -> np.ndarray: 
        # kernel = (5, 5)
        return cv2.GaussianBlur(self.image, (5, 5), sigmaX=0)

    def edge_detection(self) -> np.ndarray: 
        # Dùng canny. Nguồn: Exploring Convolutional Neural Networks for Rice Grain Classification: An Explainable AI Approach 
        image = self.image
        if np.issubdtype(image.dtype, np.floating):
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(image, threshold1=100, threshold2=200)

    def normalization(self) -> np.ndarray: 
        dst = np.empty_like(self.image, dtype=np.float32)
        return cv2.normalize(self.image, dst, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


        


