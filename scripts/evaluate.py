import argparse
import json
from pathlib import Path

from src.data import load_splits, split_fingerprint
from src.evaluation import evaluate_records, save_evaluation
from src.pipeline import RiceSVMPipeline

def evaluate(args):
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    records, class_names = load_splits(args.splits)
    pipeline = RiceSVMPipeline.load(args.model)
    if pipeline.class_names != class_names:
        raise ValueError('Dataset class mapping does not match the checkpoint.')
    expected = pipeline.training_metadata.get('split_fingerprint')
    if expected is not None and expected != split_fingerprint(records, class_names):
        raise ValueError('Evaluation manifest does not match the training manifest.')
    selected = [record for record in records if record.split == args.split]
    report, predictions = evaluate_records(pipeline, selected, args.data_root)
    report.update(model=str(args.model.resolve()), split=args.split,
                  split_fingerprint=split_fingerprint(records, class_names),
                  model_bytes=args.model.stat().st_size)
    save_evaluation(args.output, report, predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--splits', type=Path, default=Path('artifacts/splits.csv'))
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        evaluate(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
