import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .data import image_digest, read_image
from .preprocessing import SegmentationError
from .utils import write_json


def extract_records(pipeline, records, root):
    features = []
    indices = []
    rows = []
    for index, record in enumerate(records):
        if index % 5000 == 0:
            print(f'Extracting {record.split} images: {index}/{len(records)}', flush=True)
        row = {'path': record.path, 'label': record.label, 'split': record.split,
               'status': 'ok', 'error': ''}
        try:
            image = read_image(Path(root) / record.path)
            if record.sha256 and image_digest(image) != record.sha256:
                raise ValueError('Image content does not match the split manifest.')
            vector = pipeline.inspect(image)['features']
        except SegmentationError as error:
            row.update(status='segmentation_error', error=str(error))
        except ValueError as error:
            row.update(status='image_error', error=str(error))
        else:
            indices.append(index)
            features.append(vector)
            row.update(zip(pipeline.extractor.feature_names, vector.tolist()))
        rows.append(row)
    matrix = np.vstack(features) if features else np.empty((0, len(pipeline.extractor.feature_names)))
    return matrix, np.asarray(indices, dtype=int), rows


def classification_metrics(labels, predictions, class_names):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if not len(labels) or labels.shape != predictions.shape:
        raise ValueError('Nonempty aligned label and prediction arrays are required.')
    count = len(class_names)
    if np.any((labels < 0) | (labels >= count)) or np.any((predictions < -1) | (predictions >= count)):
        raise ValueError('Class indices are outside the configured mapping.')
    matrix = np.zeros((count, count + 1), dtype=int)
    for actual, predicted in zip(labels, predictions):
        matrix[actual, predicted if predicted >= 0 else count] += 1
    return {
        'n_samples': int(len(labels)), 'n_errors': int(np.count_nonzero(labels != predictions)),
        'n_failures': int(np.count_nonzero(predictions == -1)),
        'failure_rate': float(np.mean(predictions == -1)),
        'accuracy': float(accuracy_score(labels, predictions)),
        'macro_f1': float(f1_score(labels, predictions, labels=range(count), average='macro', zero_division=0)),
        'per_class': classification_report(labels, predictions, labels=list(range(count)),
                                           target_names=class_names, output_dict=True, zero_division=0),
        'confusion_matrix': matrix.tolist(),
        'confusion_rows': list(class_names), 'confusion_columns': [*class_names, 'failure'],
        'failure_policy': 'Failures count as incorrect predictions and false negatives; no samples are dropped.',
    }


def evaluate_records(pipeline, records, root):
    features, indices, rows = extract_records(pipeline, records, root)
    predictions = np.full(len(records), -1, dtype=int)
    if len(indices):
        predictions[indices] = pipeline.classifier.predict(features)
    for row, prediction in zip(rows, predictions):
        row['prediction'] = int(prediction) if prediction >= 0 else None
        row['prediction_name'] = pipeline.class_names[prediction] if prediction >= 0 else ''
        row['class_name'] = pipeline.class_names[row['label']]
    report = classification_metrics([record.label for record in records], predictions, pipeline.class_names)
    report['n_segmentation_failures'] = sum(row['status'] == 'segmentation_error' for row in rows)
    report['n_image_failures'] = sum(row['status'] == 'image_error' for row in rows)
    return report, rows


def save_rows(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty prediction or feature table.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_evaluation(output, report, rows):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'metrics.json', report)
    save_rows(output / 'predictions.csv', rows)
