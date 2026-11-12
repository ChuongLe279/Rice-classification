import preprocessing, models

class Pipeline:
    def __init__(self, config):
        self.config = config

    def load_resize_split_data(self):
        raise NotImplementedError

    def apply_aug(self):
        raise NotImplementedError

    def normalize_tensor(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError   

    def validation(self):
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