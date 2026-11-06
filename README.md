# Rice classification

## Cấu trúc

- `src/data.py`: đọc ảnh, chia dữ liệu, lưu và kiểm tra manifest.
- `src/preprocessing.py`: `RiceSegmenter`, tách một hạt hoàn chỉnh khỏi nền.
- `src/features.py`: `ShapeFeatureExtractor`, tính 16 đặc trưng hình dạng.
- `src/models.py`: `SVMClassifier`, pipeline StandardScaler và SVC.
- `src/pipeline.py`: `RiceSVMPipeline`, ghép suy luận, xem mask, lưu/nạp checkpoint.
- `src/evaluation.py`: trích đặc trưng theo danh sách ảnh, tính metric và xuất kết quả.
- `src/utils.py`: JSON, seed và các hàm đọc/đếm ảnh có sẵn.
- `scripts/`: các lệnh train, evaluate và predict.
- `configs/svm.json`: cấu hình chạy mặc định.
- `tests/test_svm_system.py`: kiểm thử hình học, dữ liệu, mô hình và lệnh chạy đầu cuối.
