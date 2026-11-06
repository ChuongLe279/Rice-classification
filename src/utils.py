import json
import random
from pathlib import Path

import cv2
import numpy as np


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def load_images_from_folder(folder):
    """Đọc các ảnh trong thư mục."""
    images = []
    labels = []
    folder = Path(folder)
    class_names = sorted(path.name for path in folder.iterdir() if path.is_dir())

    for label, class_name in enumerate(class_names):
        class_folder = folder / class_name
        for img_path in sorted(class_folder.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
                    labels.append(label)

    return images, labels, class_names

def count_image_per_class(folder):
    folder = Path(folder)
    return {
        directory.name: sum(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            and cv2.imread(str(path)) is not None
            for path in sorted(directory.iterdir())
        )
        for directory in sorted(folder.iterdir()) if directory.is_dir()
    }


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
