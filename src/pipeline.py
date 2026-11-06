import platform
from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np
import sklearn
from sklearn.metrics import f1_score
from sklearn.utils.validation import check_is_fitted

from .features import ShapeFeatureExtractor
from .models import SVMClassifier
from .preprocessing import RiceSegmenter


class RiceSVMPipeline:
    def __init__(self, class_names, segmentation=None, features=None, svm=None):
        if len(class_names) < 2 or len(set(class_names)) != len(class_names):
            raise ValueError('Provide at least two unique class names.')
        self.class_names = list(class_names)
        self.segmenter = RiceSegmenter(**(segmentation or {}))
        self.extractor = ShapeFeatureExtractor(**(features or {}))
        self.classifier = SVMClassifier(**(svm or {}))
        self.training_metadata = {}

    def inspect(self, image):
        region = self.segmenter.segment(image)
        features = self.extractor.extract(region)
        return {'mask': region.mask, 'contour': region.contour, 'features': features,
                'feature_names': list(self.extractor.feature_names),
                'component_count': region.component_count, 'area': region.area}

    def transform(self, images):
        features = [self.inspect(image)['features'] for image in images]
        if not features:
            raise ValueError('At least one image is required.')
        return np.vstack(features)

    def fit_features(self, features, labels):
        labels = np.asarray(labels)
        if set(labels.tolist()) != set(range(len(self.class_names))):
            raise ValueError('Training labels must cover every configured class index.')
        self.classifier.fit(features, labels)
        return self

    def fit(self, images, labels):
        return self.fit_features(self.transform(images), labels)

    def predict(self, images):
        return self.classifier.predict(self.transform(images))

    def select_model(self, train_features, train_labels, validation_features,
                     validation_labels, candidates, validation_indices=None):
        if not candidates:
            raise ValueError('At least one candidate is required.')
        if set(np.asarray(train_labels).tolist()) != set(range(len(self.class_names))):
            raise ValueError('Training labels must cover every configured class index.')
        best_score = -1.
        history = []
        base_config = dict(self.classifier.config)
        for candidate in candidates:
            config = {**base_config, **candidate}
            print(f'Fitting SVM: {config}', flush=True)
            classifier = SVMClassifier(**config)
            start = perf_counter()
            classifier.fit(train_features, train_labels)
            fit_seconds = perf_counter() - start
            if validation_indices is None:
                predictions = classifier.predict(validation_features)
            else:
                predictions = np.full(len(validation_labels), -1, dtype=int)
                if len(validation_indices):
                    predictions[validation_indices] = classifier.predict(validation_features)
            score = f1_score(validation_labels, predictions, average='macro',
                             labels=range(len(self.class_names)), zero_division=0)
            print(f'Fit {fit_seconds:.2f}s; validation macro-F1 {score:.6f}', flush=True)
            history.append({'config': config, 'macro_f1': float(score), 'fit_seconds': fit_seconds})
            if score > best_score:
                best_score = score
                best_classifier = classifier
        self.classifier = best_classifier
        self.training_metadata['selection'] = history
        self.training_metadata['best_validation_macro_f1'] = float(best_score)
        return history

    def save(self, path):
        check_is_fitted(self.classifier.estimator.named_steps['svm'])
        if self.classifier.estimator.n_features_in_ != len(self.extractor.feature_names):
            raise ValueError('Classifier feature dimensions do not match the image extractor.')
        path = Path(path)
        if path.exists():
            raise FileExistsError(f'Model already exists: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'format_version': 1, 'class_names': self.class_names,
            'segmentation': self.segmenter.get_config(), 'features': self.extractor.get_config(),
            'feature_names': list(self.extractor.feature_names), 'svm': self.classifier.config,
            'estimator': self.classifier.estimator, 'training_metadata': self.training_metadata,
            'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                         'opencv': cv2.__version__, 'sklearn': sklearn.__version__,
                         'joblib': joblib.__version__},
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path):
        """Load a locally trusted checkpoint created by this application."""
        payload = joblib.load(path)
        if payload['format_version'] != 1:
            raise ValueError('Unsupported checkpoint format.')
        pipeline = cls(payload['class_names'], payload['segmentation'], payload['features'], payload['svm'])
        if payload['feature_names'] != list(pipeline.extractor.feature_names):
            raise ValueError('Checkpoint feature order is incompatible.')
        pipeline.classifier.estimator = payload['estimator']
        check_is_fitted(pipeline.classifier.estimator.named_steps['svm'])
        if pipeline.classifier.estimator.n_features_in_ != len(pipeline.extractor.feature_names):
            raise ValueError('Checkpoint feature dimensions are incompatible.')
        pipeline.training_metadata = payload['training_metadata']
        return pipeline
