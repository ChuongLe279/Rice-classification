# Rice SVM implementation plan

Goal: implement the approved modular image-to-label SVM system with reproducible splits, saved inference configuration and traceable evaluation.

Architecture: data.py owns dataset records and shared manifests. preprocessing.py segments grains, features.py computes shape descriptors, models.py owns StandardScaler and SVC, and pipeline.py composes inference and persistence. evaluation.py records predictions and failures. Small scripts expose training, evaluation and prediction.

Technology: Python, NumPy, OpenCV, scikit-learn and joblib. Execute inline in the current workspace. Preserve existing AlexNet and preprocessing placeholders and the notebook. Do not create a commit. Python comments and docstrings use plain text.

1. Add behavioral tests for geometry, invalid segmentation, reproducible grouped splits, train-only scaling, validation selection and complete model round trips. Run python -m unittest discover -s tests -v and confirm missing functionality.
2. Implement src/data.py and shared JSON helpers in src/utils.py. Enumerate directories deterministically, retain duplicate-content groups in the same split, validate saved manifests and preserve the existing image-loading interface.
3. Add RiceSegmenter to src/preprocessing.py and ShapeFeatureExtractor to src/features.py. Use original-resolution uint8 images. Compute equivalent ellipse axes from region moments, raster area, contour perimeter, raster hull area and seven signed-log Hu values. Return explicit segmentation errors.
4. Implement SVMClassifier in src/models.py and RiceSVMPipeline in src/pipeline.py. Fit scaling on train only, select configurations by validation macro-F1, preserve the selected train-fitted model and save configuration, feature order, mapping and package versions with it.
5. Implement src/evaluation.py, configs/svm.json and scripts/train_svm.py, scripts/evaluate.py, scripts/predict.py. Stream image feature extraction, retain per-image failures, compute metrics over the complete evaluation split and save CSV predictions and JSON reports. Refuse accidental reuse or replacement of existing split/model outputs.
6. Add requirements.txt and README.md with environment setup, CLI commands, Python usage, color and feature conventions, duplicate-group policy, artifact descriptions and segmentation-failure metric policy.
7. Run focused unit/integration tests plus an end-to-end CLI exercise on temporary synthetic images. Verify checkpoint reload predictions, validation tuning records, failed-image accounting and absence of Markdown in Python source. Review git diff --check and git status --short.

Acceptance: shared 70/15/15 manifests with seed 42, no duplicate group spanning splits, original-resolution shape vectors, train-only normalization, validation-only selection, matching predictions after save/load and failures included as errors in evaluation. Real Rice Image Dataset results require the dataset, which is not currently in the workspace.
