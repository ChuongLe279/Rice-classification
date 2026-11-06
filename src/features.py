import cv2
import numpy as np

from .preprocessing import SegmentationError


class ShapeFeatureExtractor:
    feature_names = (
        'area', 'perimeter', 'major_axis', 'minor_axis', 'aspect_ratio',
        'circularity', 'eccentricity', 'solidity', 'extent',
        'hu_1', 'hu_2', 'hu_3', 'hu_4', 'hu_5', 'hu_6', 'hu_7',
    )

    def __init__(self, hu_epsilon=1e-30):
        if not 0 < hu_epsilon < 1:
            raise ValueError('Hu epsilon must be between zero and one.')
        self.hu_epsilon = hu_epsilon

    def get_config(self):
        return {'hu_epsilon': self.hu_epsilon}

    def extract(self, region):
        moments = cv2.moments(region.mask, binaryImage=True)
        area = moments['m00']
        perimeter = cv2.arcLength(region.contour, True)
        if area <= 0 or perimeter <= 0:
            raise SegmentationError('Degenerate grain area or contour.')
        covariance = np.array([[moments['mu20'], moments['mu11']],
                               [moments['mu11'], moments['mu02']]]) / area
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[0] <= 0:
            raise SegmentationError('Grain has a degenerate minor axis.')
        minor, major = 4 * np.sqrt(eigenvalues)
        hull = cv2.convexHull(region.contour)
        hull_mask = np.zeros_like(region.mask)
        cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
        hull_area = np.count_nonzero(hull_mask)
        _, _, width, height = cv2.boundingRect(region.contour)
        hu = cv2.HuMoments(moments).ravel()
        hu_log = -np.sign(hu) * np.log10(np.maximum(np.abs(hu), self.hu_epsilon))
        features = np.array([
            area, perimeter, major, minor, major / minor,
            4 * np.pi * area / perimeter ** 2,
            np.sqrt(max(0., 1 - (minor / major) ** 2)),
            area / hull_area, area / (width * height), *hu_log,
        ], dtype=np.float64)
        if not np.isfinite(features).all():
            raise SegmentationError('Non-finite shape features.')
        return features
