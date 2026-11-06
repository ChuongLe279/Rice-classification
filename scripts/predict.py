import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2

from src.data import read_image
from src.pipeline import RiceSVMPipeline
from src.utils import write_json


def predict(args):
    if args.output is not None and args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    pipeline = RiceSVMPipeline.load(args.model)
    image = read_image(args.image)
    start = perf_counter()
    inspected = pipeline.inspect(image)
    label = int(pipeline.classifier.predict(inspected['features'][None, :])[0])
    elapsed_ms = (perf_counter() - start) * 1000
    result = {
        'image': str(args.image), 'label': label, 'class_name': pipeline.class_names[label],
        'pipeline_ms': elapsed_ms,
        'timing_scope': 'Segmentation, shape extraction, scaling and SVM; excludes image and model loading.',
        'features': dict(zip(inspected['feature_names'], inspected['features'].tolist())),
    }
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        if not cv2.imwrite(str(args.output / 'mask.png'), inspected['mask']):
            raise OSError('Cannot save grain mask.')
        overlay = image.copy()
        cv2.drawContours(overlay, [inspected['contour']], -1, (0, 255, 0), 1)
        if not cv2.imwrite(str(args.output / 'contour.png'), overlay):
            raise OSError('Cannot save grain contour.')
        write_json(args.output / 'prediction.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        predict(args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
