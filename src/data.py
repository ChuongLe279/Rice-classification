import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from .utils import IMAGE_EXTENSIONS, read_json, write_json


GROUPING = 'sha256 of decoded BGR image shape and pixels'


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: int
    split: str
    sha256: str = ''


def image_digest(image):
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode('ascii'))
    digest.update(image.tobytes())
    return digest.hexdigest()


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def create_splits(root, seed=42, ratios=(0.70, 0.15, 0.15)):
    root = Path(root)
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError('Three positive split ratios must sum to one.')
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    if len(directories) < 2:
        raise ValueError('Dataset must contain at least two class directories.')
    class_names = [path.name for path in directories]
    rng = np.random.default_rng(seed)
    seen_labels = {}
    records = []
    for label, directory in enumerate(directories):
        print(f'Indexing class {directory.name}...', flush=True)
        groups = defaultdict(list)
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            digest = image_digest(read_image(path))
            if digest in seen_labels and seen_labels[digest] != label:
                raise ValueError(f'Identical image content has conflicting class labels: {path}')
            seen_labels[digest] = label
            groups[digest].append(path.relative_to(root).as_posix())
        if len(groups) < 3:
            raise ValueError(f'Class {directory.name} needs at least three distinct image groups.')
        print(f'Indexed {sum(map(len, groups.values()))} images in {len(groups)} content groups.', flush=True)
        keys = list(groups)
        rng.shuffle(keys)
        keys.sort(key=lambda key: -len(groups[key]))
        targets = np.asarray(ratios) * sum(map(len, groups.values()))
        counts = np.zeros(3, dtype=int)
        for index, digest in enumerate(keys):
            empty = np.flatnonzero(counts == 0)
            if len(keys) - index == len(empty):
                split_index = int(empty[0])
            else:
                split_index = int(np.argmax(targets - counts))
            split = ('train', 'validation', 'test')[split_index]
            records.extend(ImageRecord(path, label, split, digest) for path in groups[digest])
            counts[split_index] += len(groups[digest])
    return sorted(records, key=lambda record: record.path), class_names


def validate_records(records, class_names):
    if (len(class_names) < 2 or any(not isinstance(name, str) or not name for name in class_names)
            or len(set(class_names)) != len(class_names)):
        raise ValueError('Class names must be unique and contain at least two classes.')
    paths = set()
    digests = {}
    for record in records:
        path = PurePosixPath(record.path)
        if path.is_absolute() or '..' in path.parts or '\\' in record.path or ':' in record.path:
            raise ValueError(f'Manifest path must be relative to dataset root: {record.path}')
        if record.path in paths:
            raise ValueError(f'Duplicate manifest path: {record.path}')
        paths.add(record.path)
        if record.split not in {'train', 'validation', 'test'} or not 0 <= record.label < len(class_names):
            raise ValueError(f'Invalid split or label: {record}')
        if len(path.parts) != 2 or path.parts[0] != class_names[record.label]:
            raise ValueError(f'Image directory does not match its class label: {record.path}')
        if not re.fullmatch(r'[0-9a-f]{64}', record.sha256):
            raise ValueError(f'Missing or invalid content hash: {record.path}')
        assignment = (record.label, record.split)
        if record.sha256 in digests and digests[record.sha256] != assignment:
            raise ValueError('Duplicate content crosses splits or class labels.')
        digests[record.sha256] = assignment
    for split in ('train', 'validation', 'test'):
        labels = {record.label for record in records if record.split == split}
        if labels != set(range(len(class_names))):
            raise ValueError(f'Every class must appear in {split}.')


def save_splits(records, class_names, path, seed=42, ratios=(0.70, 0.15, 0.15)):
    validate_records(records, class_names)
    path = Path(path)
    metadata_path = path.with_suffix('.json')
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f'Split manifest already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'label', 'split', 'sha256'])
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    write_json(metadata_path, {'format_version': 1, 'class_names': class_names, 'seed': seed,
                              'requested_ratios': list(ratios), 'grouping': GROUPING})


def load_splits(path):
    path = Path(path)
    metadata = read_json(path.with_suffix('.json'))
    if (metadata.get('format_version') != 1 or metadata.get('grouping') != GROUPING
            or type(metadata.get('seed')) is not int or not 0 <= metadata['seed'] < 2 ** 32):
        raise ValueError('Invalid split metadata format, grouping policy or seed.')
    ratios = metadata.get('requested_ratios')
    if (not isinstance(ratios, list) or len(ratios) != 3
            or any(type(value) not in (int, float) or not 0 < value < 1 for value in ratios)
            or not np.isclose(sum(ratios), 1)):
        raise ValueError('Invalid requested split ratios in metadata.')
    class_names = metadata['class_names']
    with path.open(encoding='utf-8', newline='') as stream:
        records = [ImageRecord(row['path'], int(row['label']), row['split'], row['sha256'])
                   for row in csv.DictReader(stream)]
    validate_records(records, class_names)
    return records, class_names


def split_fingerprint(records, class_names):
    payload = {'class_names': class_names,
               'records': [asdict(record) for record in sorted(records, key=lambda item: item.path)]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()


def verify_dataset(records, root):
    for record in records:
        image = read_image(Path(root) / record.path)
        if not record.sha256 or image_digest(image) != record.sha256:
            raise ValueError(f'Image does not match split manifest: {record.path}')
