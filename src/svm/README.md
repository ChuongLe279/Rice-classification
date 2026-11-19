# Shape + SVM

Phần SVM độc lập trong cấu trúc mới của repo. Chạy các lệnh từ thư mục gốc `rice_classification`.

## Cấu trúc

- `svm.py`: điểm chạy train/evaluate/predict/download.
- `models.py`: StandardScaler và SVC.
- `segmentation.py`: tách mask/contour và kiểm tra một hạt hoàn chỉnh.
- `features.py`: tính 16 đặc trưng hình dạng.
- `pipeline.py`: ghép ảnh đầu vào tới nhãn, lưu/nạp checkpoint.
- `utils.py`: đọc ảnh, chia tập, hash, manifest và JSON.
- `evaluation.py`: trích đặc trưng cả tập, tính metric, ghi CSV.
- `train.py`, `evaluate.py`, `predict.py`: luồng chạy riêng cho SVM.
- `download_dataset.py`: tải Rice Image Dataset từ Kaggle.
- `config.json`, `requirements.txt`, `tests/`: cấu hình, thư viện và kiểm thử.


## Cài thư viện

```powershell
python -m pip install -r src/svm/requirements.txt
```

## Dataset

Trên máy local hiện có 75.000 ảnh ở `data/Rice_Image_Dataset`, với các thư mục `Arborio`, `Basmati`, `Ipsala`, `Jasmine`, `Karacadag`.

Máy mới có thể tải bằng:

```powershell
python -m src.svm.svm download --output data/rice
```

## Huấn luyện

```powershell
python -m src.svm.svm train --data-root "data/Rice_Image_Dataset" --splits "artifacts/splits.csv" --output "artifacts/results/svm_rebuild_train_01"
```


## Đánh giá checkpoint local

```powershell
python -m src.svm.svm evaluate --data-root "data/Rice_Image_Dataset" --splits "artifacts/splits.csv" --model "artifacts/results/svm_local_01/model.joblib" --output "artifacts/results/svm_rebuild_test_01"
```

## Dự đoán

```powershell
python -m src.svm.svm predict --model "artifacts/results/svm_local_01/model.joblib" --image "data/Rice_Image_Dataset/Arborio/Arborio (100).jpg"
```

## Quy ước

Đầu vào uint8 BGR từ OpenCV hoặc grayscale, tối thiểu 8 pixel mỗi chiều. Giữ ảnh gốc, không resize/normalize theo AlexNet trước khi đưa vào SVM. Diện tích đo bằng pixel vuông, chiều dài bằng pixel.

Mask được threshold, opening/closing, chọn component chính và lấp lỗ bên trong contour. Báo lỗi nếu không có hạt, nhiều vùng đáng kể hoặc hạt chạm biên. `SegmentationResult` chỉ đóng gói kết quả, không phải mô hình học.

Đặc trưng: diện tích raster, chu vi contour, hai trục từ covariance của region moments, aspect ratio, circularity, eccentricity, solidity raster, extent bounding box thẳng và 7 Hu moments biến đổi log có dấu. Thứ tự cố định ở `ShapeFeatureExtractor.feature_names`. Extent phụ thuộc góc xoay; các đặc trưng kích thước phụ thuộc tỉ lệ ảnh.

## Kiểm thử

```powershell
python -m unittest discover -s src/svm/tests -v
```
