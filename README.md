# INS3155 – Computer Vision Midterm Project

Traditional Image Processing for Filtering, 3D Reconstruction, and Image Stitching.

- Sinh viên: <Lương Hải Yến> – MSSV: <23070019>`

## 1. Môi trường

- Python 3.13.5
- OpenCV 5.0.0 (`opencv-contrib-python`), NumPy, SciPy, Matplotlib, scikit-image, Open3D 0.20.0, pandas, plotly

```bash
python -m venv cv_env
cv_env\Scripts\activate          # Windows (macOS/Linux: source cv_env/bin/activate)
pip install -r requirements.txt
```

## 2. Cấu trúc thư mục

```
StudentID_FullName_Code/
├── README.md
├── requirements.txt
├── data/
│   ├── partA/      # ảnh sạch cho lọc ảnh
│   ├── partB/      # cặp stereo (+ calib, ground truth)
│   └── partC/      # 4 ảnh chồng lấp
├── src/
│   ├── utils.py
│   ├── part_a_filtering.py
│   ├── part_b_stereo.py
│   └── part_c_stitching.py
└── results/        # ảnh và CSV sinh ra bởi các script
```

## 3. Cách chạy

Chạy từ thư mục gốc của project:

```bash
python src/part_a_filtering.py
python src/part_b_stereo.py
python src/part_c_stitching.py
```

Kết quả (hình, bảng CSV, point cloud `.ply`) nằm trong `results/partA`, `results/partB`, `results/partC`.

## 4. Dữ liệu

- Part A: `<mô tả ảnh>`
- Part B: `<tên cặp stereo, nguồn: Middlebury Stereo>`
- Part C: `<ảnh tự chụp / nguồn>`

## 5. Ghi chú

- Seed cố định (`utils.SEED = 42`) để kết quả tái lập được.
- Part C so sánh SIFT và ORB (ratio test 0.75, nhiều ngưỡng RANSAC).
- `<Ghi chú khác: tham số quan trọng, giới hạn, lỗi đã biết>`
