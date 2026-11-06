import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src import models, preprocessing, utils


def grain(axes=(34, 12), angle=0, center=(64, 64)):
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.ellipse(image, center, axes, angle, 0, 360, (220, 220, 220), -1)
    return image


class GeometryTests(unittest.TestCase):
    def test_region_geometry_and_rotation_stability(self):
        from src.features import ShapeFeatureExtractor

        segmenter = preprocessing.RiceSegmenter()
        extractor = ShapeFeatureExtractor()
        first = extractor.extract(segmenter.segment(grain()))
        rotated = extractor.extract(segmenter.segment(grain(angle=65)))
        values = dict(zip(extractor.feature_names, first))
        self.assertEqual(first.shape, (16,))
        self.assertTrue(np.isfinite(first).all())
        self.assertAlmostEqual(values['aspect_ratio'], 34 / 12, delta=0.2)
        self.assertGreater(values['solidity'], 0.95)
        self.assertLessEqual(values['solidity'], 1)
        np.testing.assert_allclose(first[:8], rotated[:8], rtol=0.06)

    def test_empty_multiple_and_clipped_grains_raise(self):
        segmenter = preprocessing.RiceSegmenter()
        images = [np.zeros((128, 128, 3), np.uint8), grain(center=(0, 64))]
        multiple = np.zeros((128, 128, 3), np.uint8)
        cv2.circle(multiple, (28, 64), 12, (255, 255, 255), -1)
        cv2.circle(multiple, (95, 64), 12, (255, 255, 255), -1)
        images.append(multiple)
        for image in images:
            with self.subTest(image_sum=int(image.sum())):
                with self.assertRaises(preprocessing.SegmentationError):
                    segmenter.segment(image)

    def test_morphology_cannot_hide_original_boundary_contact(self):
        image = grain()
        cv2.line(image, (0, 64), (30, 64), (220, 220, 220), 1)
        with self.assertRaises(preprocessing.SegmentationError):
            preprocessing.RiceSegmenter().segment(image)


class ModelTests(unittest.TestCase):
    def test_scaler_is_fit_on_train_only_during_selection(self):
        from src.pipeline import RiceSVMPipeline

        train = np.array([[0., 1.], [1., 1.], [4., 2.], [5., 2.]])
        validation = np.array([[100., 1.], [105., 2.]])
        pipeline = RiceSVMPipeline(class_names=['round', 'long'])
        history = pipeline.select_model(
            train, [0, 0, 1, 1], validation, [0, 1],
            [{'C': 0.1}, {'C': 1.0}, {'C': 10.0}],
        )
        np.testing.assert_allclose(pipeline.classifier.estimator.named_steps['scaler'].mean_, train.mean(axis=0))
        self.assertEqual(len(history), 3)
        self.assertTrue(all('fit_seconds' in item and 'macro_f1' in item for item in history))

    def test_image_pipeline_round_trip_and_failure_accounting(self):
        from src.data import ImageRecord
        from src.evaluation import evaluate_records
        from src.pipeline import RiceSVMPipeline

        images = [grain((25, 20)), grain((26, 20)), grain((40, 9)), grain((42, 9))]
        pipeline = RiceSVMPipeline(class_names=['round', 'long'])
        pipeline.fit(images, [0, 0, 1, 1])
        expected = pipeline.predict(images)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'svm.joblib'
            pipeline.save(checkpoint)
            loaded = RiceSVMPipeline.load(checkpoint)
            np.testing.assert_array_equal(loaded.predict(images), expected)
            self.assertEqual(loaded.class_names, ['round', 'long'])
            self.assertEqual(loaded.inspect(images[0])['mask'].shape, (128, 128))
            cv2.imwrite(str(root / 'good.png'), images[0])
            cv2.imwrite(str(root / 'empty.png'), np.zeros_like(images[0]))
            records = [ImageRecord('good.png', 0, 'test'), ImageRecord('empty.png', 1, 'test')]
            report, predictions = evaluate_records(loaded, records, root)
            self.assertEqual(report['n_samples'], 2)
            self.assertEqual(report['n_failures'], 1)
            self.assertEqual(report['n_errors'], 1)
            self.assertEqual(report['accuracy'], 0.5)
            self.assertEqual(sum(sum(row) for row in report['confusion_matrix']), 2)
            self.assertEqual(predictions[1]['status'], 'segmentation_error')
            self.assertIsNone(predictions[1]['prediction'])

    def test_changed_image_is_recorded_as_failure_against_manifest_hash(self):
        from src.data import ImageRecord, image_digest
        from src.evaluation import evaluate_records
        from src.pipeline import RiceSVMPipeline

        pipeline = RiceSVMPipeline(['round', 'long'])
        pipeline.fit([grain((25, 20)), grain((40, 9))], [0, 1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = grain((25, 20))
            cv2.imwrite(str(root / 'changed.png'), grain((40, 9)))
            record = ImageRecord('changed.png', 0, 'test', image_digest(original))
            report, rows = evaluate_records(pipeline, [record], root)
            self.assertEqual(report['n_failures'], 1)
            self.assertEqual(rows[0]['status'], 'image_error')
            self.assertIn('manifest', rows[0]['error'])


class DataTests(unittest.TestCase):
    def test_manifest_rejects_wrong_directory_label_and_missing_hash(self):
        from src.data import ImageRecord, validate_records

        names = ['long', 'round']
        records = [ImageRecord(f'{name}/{split}.png', label, split, f'{label * 3 + index:064x}')
                   for label, name in enumerate(names)
                   for index, split in enumerate(('train', 'validation', 'test'))]
        validate_records(records, names)
        records[0] = ImageRecord('round/train.png', 0, 'train', 'f' * 64)
        with self.assertRaises(ValueError):
            validate_records(records, names)
        records[0] = ImageRecord('long/train.png', 0, 'train')
        with self.assertRaises(ValueError):
            validate_records(records, names)

    def test_split_round_trip_is_deterministic_and_keeps_duplicates_together(self):
        from src.data import create_splits, load_splits, save_splits

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['long', 'round']:
                folder = root / name
                folder.mkdir()
                for index in range(20):
                    image = grain((32, 10 + index % 4), angle=index * 7)
                    if name == 'long':
                        image[image > 0] = 180
                    cv2.imwrite(str(folder / f'{index}.png'), image)
                duplicate = grain((32, 10), angle=0)
                if name == 'long':
                    duplicate[duplicate > 0] = 180
                cv2.imwrite(str(folder / 'duplicate.png'), duplicate)
            first, names = create_splits(root)
            second, second_names = create_splits(root)
            self.assertEqual(first, second)
            self.assertEqual(names, second_names)
            self.assertEqual(names, ['long', 'round'])
            grouped = {}
            for record in first:
                grouped.setdefault(record.sha256, set()).add(record.split)
            self.assertTrue(all(len(splits) == 1 for splits in grouped.values()))
            manifest = root / 'splits.csv'
            save_splits(first, names, manifest)
            restored, restored_names = load_splits(manifest)
            self.assertEqual(restored, first)
            self.assertEqual(restored_names, names)
            self.assertEqual(len(first), 42)

    def test_loader_ignores_files_at_dataset_root_without_label_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata.txt').write_text('dataset', encoding='utf-8')
            (root / 'rice').mkdir()
            cv2.imwrite(str(root / 'rice' / 'sample.png'), grain())
            images, labels, names = utils.load_images_from_folder(root)
            self.assertEqual(names, ['rice'])
            self.assertEqual(labels, [0])
            self.assertEqual(len(images), 1)
            self.assertEqual(utils.count_image_per_class(root), {'rice': 1})


class CommandTests(unittest.TestCase):
    def test_train_evaluate_predict_commands_share_checkpoint_and_split(self):
        workspace = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'data'
            for name, axes in [('round', (24, 20)), ('long', (40, 9))]:
                folder = dataset / name
                folder.mkdir(parents=True)
                for index in range(30):
                    cv2.imwrite(str(folder / f'{index}.png'), grain(axes, angle=index * 6))
            manifest = root / 'splits.csv'
            output = root / 'run'

            def run(module, *args, expected=0):
                result = subprocess.run(
                    [sys.executable, '-m', module, *map(str, args)],
                    cwd=workspace, text=True, encoding='utf-8', capture_output=True,
                )
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                return result

            run('scripts.train_svm', '--data-root', dataset, '--splits', manifest, '--output', output)
            checkpoint = output / 'model.joblib'
            self.assertTrue(checkpoint.exists())
            metadata = json.loads((output / 'run.json').read_text(encoding='utf-8'))
            self.assertEqual(len(metadata['selection']), 3)
            self.assertEqual(metadata['split_counts']['train'], 42)
            self.assertEqual(sum(metadata['split_counts'].values()), 60)
            with (output / 'features.csv').open(encoding='utf-8', newline='') as stream:
                self.assertNotIn('test', {row['split'] for row in csv.DictReader(stream)})
            evaluation = root / 'test-results'
            run('scripts.evaluate', '--data-root', dataset, '--splits', manifest,
                '--model', checkpoint, '--output', evaluation)
            metrics = json.loads((evaluation / 'metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(metrics['n_samples'], metadata['split_counts']['test'])
            self.assertGreaterEqual(metrics['accuracy'], 0.8)
            prediction = root / 'prediction'
            run('scripts.predict', '--model', checkpoint, '--image', dataset / 'long' / '0.png',
                '--output', prediction)
            result = json.loads((prediction / 'prediction.json').read_text(encoding='utf-8'))
            self.assertEqual(result['class_name'], 'long')
            self.assertTrue((prediction / 'mask.png').exists())
            run('scripts.train_svm', '--data-root', dataset, '--splits', manifest,
                '--output', output, expected=2)
            (root / 'bad.png').write_bytes(b'invalid image')
            failure = run('scripts.predict', '--model', checkpoint, '--image', root / 'bad.png', expected=2)
            self.assertIn('Cannot read image', failure.stderr)


if __name__ == '__main__':
    unittest.main()
