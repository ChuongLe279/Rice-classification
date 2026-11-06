import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from zipfile import ZipFile

from src.utils import write_json


DATASET = 'muratkokludataset/rice-image-dataset'
EXPECTED_CLASSES = {'Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag'}


def download(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Dataset output already exists: {output}')
    output.mkdir(parents=True)
    archive = output / 'rice-image-dataset.zip.part'
    digest = hashlib.sha256()
    url = f'https://www.kaggle.com/api/v1/datasets/download/{DATASET}'
    request = Request(url, headers={'User-Agent': 'rice-classification/1.0'})
    size = 0
    with urlopen(request, timeout=120) as response, archive.open('wb') as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    counts = {name: 0 for name in sorted(EXPECTED_CLASSES)}
    unpacked = 0
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            path = PurePosixPath(member.filename)
            if member.is_dir() or len(path.parts) < 2:
                continue
            class_name = path.parts[-2]
            if class_name not in EXPECTED_CLASSES or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
                raise ValueError('Unsafe archive member path.')
            destination = output / 'Rice_Image_Dataset' / class_name / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, destination.open('xb') as target:
                shutil.copyfileobj(source, target)
            counts[class_name] += 1
            unpacked += member.file_size
    if any(count != 15000 for count in counts.values()):
        raise ValueError(f'Unexpected dataset counts: {counts}')
    metadata = {
        'source': f'https://www.kaggle.com/datasets/{DATASET}',
        'download_url': url, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
        'archive_sha256': digest.hexdigest(), 'archive_bytes': size,
        'extracted_bytes': unpacked, 'class_counts': counts,
        'dataset_root': str(output / 'Rice_Image_Dataset'),
    }
    write_json(output / 'download.json', metadata)
    archive.unlink()
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Download the public Rice Image Dataset from Kaggle.')
    parser.add_argument('--output', type=Path, default=Path('data/rice'))
    args = parser.parse_args()
    try:
        download(args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
