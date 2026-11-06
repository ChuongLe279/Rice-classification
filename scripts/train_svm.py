import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from src.data import create_splits, load_splits, save_splits, split_fingerprint, verify_dataset
from src.evaluation import evaluate_records, extract_records, save_rows
from src.pipeline import RiceSVMPipeline
from src.utils import read_json, set_seed, write_json


def train(args):
    started = perf_counter()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new run directory: {output}')
    config = read_json(args.config)
    set_seed(config['seed'])
    manifest = Path(args.splits)
    if manifest.exists():
        records, class_names = load_splits(manifest)
        verify_dataset(records, args.data_root)
    else:
        records, class_names = create_splits(args.data_root, seed=config['seed'])
        save_splits(records, class_names, manifest, seed=config['seed'])
    pipeline = RiceSVMPipeline(class_names, config['segmentation'], config['features'], config['svm'])
    train_records = [record for record in records if record.split == 'train']
    validation_records = [record for record in records if record.split == 'validation']
    train_features, train_indices, train_rows = extract_records(pipeline, train_records, args.data_root)
    validation_features, validation_indices, validation_rows = extract_records(
        pipeline, validation_records, args.data_root)
    train_labels = np.asarray([record.label for record in train_records])[train_indices]
    validation_labels = [record.label for record in validation_records]
    history = pipeline.select_model(train_features, train_labels, validation_features,
                                    validation_labels, config['candidates'], validation_indices)
    metadata = {
        'config': config, 'class_names': class_names, 'splits': str(manifest.resolve()),
        'split_fingerprint': split_fingerprint(records, class_names),
        'split_counts': dict(Counter(record.split for record in records)),
        'train_usable': int(len(train_indices)), 'train_failures': len(train_records) - len(train_indices),
        'validation_usable': int(len(validation_indices)),
        'validation_failures': len(validation_records) - len(validation_indices),
        'selection': history, 'selected_svm': pipeline.classifier.config,
        'feature_names': list(pipeline.extractor.feature_names),
        'selection_policy': 'Highest validation macro-F1; first candidate wins ties; no train-validation refit.',
        'elapsed_seconds': perf_counter() - started,
    }
    pipeline.training_metadata.update(metadata)
    output.mkdir(parents=True, exist_ok=False)
    pipeline.save(output / 'model.joblib')
    save_rows(output / 'features.csv', train_rows + validation_rows)
    report, predictions = evaluate_records(pipeline, validation_records, args.data_root)
    write_json(output / 'validation_metrics.json', report)
    save_rows(output / 'validation_predictions.csv', predictions)
    write_json(output / 'run.json', metadata)
    print(json.dumps({'model': str(output / 'model.joblib'), 'selected_svm': pipeline.classifier.config,
                      'validation_macro_f1': report['macro_f1'],
                      'train_failures': metadata['train_failures'],
                      'validation_failures': report['n_failures']}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs' / 'svm.json')
    parser.add_argument('--splits', type=Path, default=Path('artifacts/splits.csv'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/results/svm'))
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
