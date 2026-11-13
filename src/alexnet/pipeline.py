import preprocessing, models
from utils import *
from sklearn.model_selection import train_test_split


DATASET_PATH = ""

class Pipeline:
    def __init__(self, config):
        self.config = config

    def load_resize_split_data(self):
        # load ảnh
        images, labels, class_names = load_images_from_folder(DATASET_PATH)

        # resize ảnh
        resized_images = []
        for image in images:
            resized_image  = preprocessing.Preprocessing.resize(image, width=10, height=10)
            resized_images.append(resized_image)

        # split test/train
        X_temp, X_test, y_temp, y_test = train_test_split(
            resized_images, labels,
            test_size=0.15,
            stratify=labels,
            random_state=42,
        )
        
        # split train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.15 / 0.85,
            stratify=y_temp,
            random_state=42,
        )
        

    def apply_aug(self):
        raise NotImplementedError

    def normalize_tensor(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError   

    def predict(self):
        raise NotImplementedError
    
    def evaluate(self):
        raise NotImplementedError

    def run(self):
        self.load_resize_split_data()
        self.apply_aug()
        self.normalize_tensor()
        self.load_model()
        self.run()
        self.predict()
        self.evaluate()