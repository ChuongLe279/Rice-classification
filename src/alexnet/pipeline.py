import alexnet.tasks as tasks, models
from utils import *
from sklearn.model_selection import train_test_split


DATASET_PATH = ""

class AlexNetPipeline:
    def __init__(self, config):
        self.config = config

    def split_data(self):
        """split train -> train / val: 80 / 20"""
        

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
        self.split_data()
        self.apply_aug()
        self.normalize_tensor()
        self.load_model()
        self.run()
        self.predict()
        self.evaluate()