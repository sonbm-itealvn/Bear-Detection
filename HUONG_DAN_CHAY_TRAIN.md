# HƯỚNG DẪN CHI TIẾT CÁCH CHẠY HUẤN LUYỆN (TRAINING GUIDE)
## HỆ THỐNG PHÁT HIỆN GẤU CHO CHIP NOVATEK NT98560

---

## 1. TỔNG QUAN CÁC BÀI TOÁN HUẤN LUYỆN TRONG DỰ ÁN

Trong codebase `Bear Detection`, có **3 quy trình huấn luyện độc lập** phục vụ từng mục đích cụ thể:

| Bài toán | File mã nguồn | Mô hình / Kiến trúc | Mục đích |
| :--- | :--- | :--- | :--- |
| **1. Backbone Phân loại (Chính)** | `train_mobilenetv2_relu_novaic_23_4.py`<br>`train_v18.sh` | **RegNetX-800MF** (V18/V19) hoặc **MobileNetV2-ReLU** | Phân loại nhị phân `bear` vs `non_bear`, triệt tiêu FP, chịu được lượng tử hóa NPU Novatek. |
| **2. Bounding Box Head (det/)** | `det/train_det.py` | **V19 Frozen + DetHead (CenterNet/TTFNet)** | Giữ nguyên bộ phân loại V19, huấn luyện thêm đầu dự đoán tọa độ hộp gấu (Stride 16). |
| **3. Mô hình YOLO Thích ứng** | `det/yolo/train_yolo.py` | **YOLOv8n ReLU** (640x384) | Biến thể YOLO siêu nhẹ (16.36 ms trên NPU) có tầng Adapter Conv 1x1 BGR->RGB. |

---

## 2. KỊCH BẢN 1: CHẠY THỬ NGHIỆM NHANH (SMOKE TEST / DRY RUN)
*(Thích hợp để chạy ngay trên máy cá nhân, máy tính không có GPU hoặc muốn kiểm tra code có lỗi không)*

### Bước 2.1: Chuẩn bị môi trường Python
> [!IMPORTANT]
> PyTorch và các thư viện thị giác hiện hoạt động ổn định nhất trên **Python 3.10 hoặc 3.11**. Khuyến nghị tạo một môi trường ảo riêng biệt:

```bash
# Tạo môi trường ảo với Python 3.10 hoặc 3.11
py -3.10 -m venv venv_bear
# Hoặc trên Linux/macOS: python3 -m venv venv_bear

# Kích hoạt môi trường (Windows PowerShell):
.\venv_bear\Scripts\Activate.ps1
# (Trên Windows CMD: .\venv_bear\Scripts\activate.bat)
# (Trên Linux: source venv_bear/bin/activate)

# Cài đặt PyTorch và các thư viện tối thiểu:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# (Nếu máy không có GPU NVIDIA, cài bản CPU: pip install torch torchvision)

# Cài các gói cần thiết:
pip install timm pillow numpy scipy opencv-python tqdm
```

---

### Bước 2.2: Chạy thử tự động chỉ với 1 lệnh (`quick_smoke_train.py`)
Tôi đã tạo sẵn cho bạn script **[`quick_smoke_train.py`](file:///D:/Bear%20Detection/Bear%20Detection/quick_smoke_train.py)**. Script này sẽ:
1. Tự động sinh ra một tập dữ liệu giả lập siêu nhỏ gồm ảnh kích thước chuẩn $640 \times 360$ trong thư mục `dummy_smoke_dataset/`.
2. Thiết lập cấu hình chạy 1 epoch thử nghiệm, kích thước batch = 2, không cần GPU, tắt xác thực AWS Bedrock.
3. Chạy kiểm thử toàn bộ pipeline và báo cáo kết quả.

**Lệnh thực thi:**
```bash
python quick_smoke_train.py
```

---

### Bước 2.3: Tự chạy lệnh thủ công với các cờ ghi đè (Override Flags)
Nếu bạn muốn tự truyền các tham số tùy chỉnh:

```bash
# Trên Windows PowerShell:
$env:OVERRIDE_EPOCHS="1"
$env:OVERRIDE_BATCH_SIZE="2"
$env:OVERRIDE_NUM_WORKERS="0"
$env:OVERRIDE_HEM_START="999"
$env:OVERRIDE_RUN_DIR="my_test_run"

python train_mobilenetv2_relu_novaic_23_4.py `
    --data-dir "dummy_smoke_dataset" `
    --no-claude-verify

# Trên Linux / macOS Bash:
OVERRIDE_EPOCHS=1 OVERRIDE_BATCH_SIZE=2 OVERRIDE_NUM_WORKERS=0 OVERRIDE_HEM_START=999 OVERRIDE_RUN_DIR=my_test_run \
    python train_mobilenetv2_relu_novaic_23_4.py \
    --data-dir dummy_smoke_dataset \
    --no-claude-verify
```

---

## 3. KỊCH BẢN 2: HUẤN LUYỆN TOÀN DIỆN TRÊN SERVER GPU
*(Dành cho huấn luyện thật trên máy trạm GPU, Linux Cluster, Google Colab hoặc Kaggle)*

### Bước 3.1: Chuẩn bị Cấu trúc Dữ liệu Chuẩn
Thư mục dataset (mặc định tên là `bear_dataset_v_final_final`) phải có cấu trúc như sau:

```text
bear_dataset_v_final_final/
├── train/
│   ├── bear/         # Chứa hàng chục ngàn ảnh gấu huấn luyện
│   └── non_bear/     # Chứa ảnh không phải gấu (người, nai, lợn rừng, chó, cảnh rừng)
└── val/
    ├── bear/         # Ảnh gấu kiểm thử
    └── non_bear/     # Ảnh không phải gấu kiểm thử
```

---

### Bước 3.2: Lệnh Huấn luyện Backbone V19 (RegNetX-800MF)
Đây là cấu hình đã tạo ra checkpoint V19 nghiệm thu thành công với đối tác:

```bash
# Cách 1: Dùng script có sẵn train_v18.sh (chỉ định GPU, ví dụ GPU 0,1)
chmod +x train_v18.sh
./train_v18.sh 0,1 BearDetection_v19_regnetx

# Cách 2: Chạy trực tiếp bằng Python:
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OVERRIDE_BACKBONE=regnetx_008 \
OVERRIDE_CUDA_DEVICES="0,1" \
OVERRIDE_RUN_DIR="BearDetection_v19_regnetx" \
OVERRIDE_BATCH_SIZE=32 \
OVERRIDE_NUM_WORKERS=8 \
OVERRIDE_EPOCHS=100 \
OVERRIDE_PATIENCE=15 \
OVERRIDE_LR=0.001 \
OVERRIDE_HEM_CLEAN=1 \
OVERRIDE_HEM_WORKERS=8 \
python -u train_mobilenetv2_relu_novaic_23_4.py \
    --data-dir bear_dataset_v_final_final \
    --cache-only-verify \
    2>&1 | tee train_v19.log
```

> [!TIP]
> - Nếu bạn **không có tài khoản AWS Bedrock Claude**, hãy truyền cờ `--no-claude-verify` (hoặc `--cache-only-verify` nếu đã có sẵn file cache `hem_gemini_cache.pkl`).
> - Nếu muốn huấn luyện MobileNetV2 thay vì RegNetX, chỉ cần bỏ biến `OVERRIDE_BACKBONE=regnetx_008`.

---

### Bước 3.3: Lệnh Huấn luyện Box Detection Head (`det/`)
Sau khi đã có checkpoint bộ phân loại V19 (ví dụ: `checkpoint_epoch58.pth`), ta đóng băng backbone và chỉ huấn luyện đầu dò tọa độ:

```bash
# Huấn luyện DetHead Stride 16 trên GPU 0:
ALLOW_ALL_GPUS=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python -m det.train_det \
    --stride 16 \
    --up up_slice \
    --ch 128 \
    --gpu 0 \
    --batch 32 \
    --epochs 24 \
    --lr 0.001 \
    --workers 8 \
    --root bear_dataset_v_final_final \
    --store det_labels/labels_v1.jsonl \
    --cache det_labels/sha1_cache.json \
    --eval-set det_labels/eval_set.json \
    --out det_runs/s16_v2
```

---

### Bước 3.4: Lệnh Huấn luyện YOLOv8n ReLU thích ứng
Nếu muốn huấn luyện nhánh YOLO:

```bash
# Bước 1: Xuất dữ liệu sang định dạng YOLO (với vùng ignore được phủ xám 114)
python -m det.yolo.export_dataset \
    --store det_labels/labels_v1.jsonl \
    --root bear_dataset_v_final_final \
    --out det_runs/yolo_dataset

# Bước 2: Huấn luyện YOLOv8n với kích thước 384x640:
python -m det.yolo.train_yolo \
    --data det_runs/yolo_dataset/dataset.yaml \
    --epochs 60 \
    --batch 32 \
    --imgsz 640 384 \
    --device 0 \
    --project det_runs \
    --name yolo_v8n_v2
```

---

## 4. BẢNG TRA CỨU CÁC THAM SỐ VÀ BIẾN MÔI TRƯỜNG

### Các biến môi trường ghi đè (`OVERRIDE_*`)

| Tên biến | Kiểu dữ liệu | Mặc định | Ý nghĩa & Khuyến nghị |
| :--- | :--- | :--- | :--- |
| `OVERRIDE_BACKBONE` | String | `None` (MobileNetV2) | Đặt thành `regnetx_008` để dùng RegNetX-800MF (Bản V19 chuẩn). |
| `OVERRIDE_CUDA_DEVICES`| String | `""` (hoặc auto) | Chỉ định ID GPU (ví dụ `"0"`, `"0,1"`). |
| `OVERRIDE_RUN_DIR` | String | `BearDetection_...` | Tên thư mục lưu kết quả, checkpoint và log. |
| `OVERRIDE_BATCH_SIZE` | Int | `32` | Kích thước batch cho mỗi GPU (chạy thử nghiệm đặt `2` hoặc `4`). |
| `OVERRIDE_NUM_WORKERS` | Int | `8` | Số luồng tải dữ liệu (trên Windows hoặc máy yếu đặt `0`). |
| `OVERRIDE_EPOCHS` | Int | `100` | Số epoch tối đa (chạy thử nghiệm đặt `1`). |
| `OVERRIDE_PATIENCE` | Int | `15` | Số epoch dừng sớm (Early Stopping) nếu không tiến bộ. |
| `OVERRIDE_LR` | Float | `0.001` | Tốc độ học ban đầu của optimizer AdamW/SGD. |
| `OVERRIDE_HEM_START` | Int | `5` | Epoch bắt đầu chạy Hard Example Mining (đặt `999` để tắt HEM khi test nhanh). |
| `OVERRIDE_TRAIN_FRACTION`| Float | `1.0` | Tỉ lệ tập train cần học (ví dụ `0.05` để chỉ học trên 5% ảnh, test nhanh). |

---

### Các cờ dòng lệnh (CLI Flags)

| Cờ lệnh | Ý nghĩa |
| :--- | :--- |
| `--data-dir <path>` | Đường dẫn tới thư mục dataset chứa `train/` và `val/`. |
| `--no-claude-verify` | **Rất quan trọng:** Tắt hoàn toàn việc gọi API AWS Bedrock Claude khi phát hiện mẫu khó. |
| `--cache-only-verify` | Chỉ dùng kết quả phán đoán đã có trong file cache `hem_gemini_cache.pkl`, không gọi API mạng. |
| `--resume <path>` | Tiếp tục huấn luyện từ checkpoint dở dang (ví dụ: `checkpoint_epoch20.pth`). |
| `--supcon` | Bật thêm nhánh chiếu tương phản Supervised Contrastive Learning (SupCon). |

---

## 5. CÁCH THEO DÕI VÀ KIỂM TRA ĐẦU RA

Sau khi chạy xong, thư mục đầu ra (ví dụ: `BearDetection_v19_regnetx/` hoặc `my_test_run/`) sẽ có các file:

1. **`models/best_model.pth`:** Trọng số của epoch đạt điểm F1 / Accuracy cao nhất.
2. **`checkpoints/checkpoint_epoch<N>.pth`:** Checkpoint lưu lại đầy đủ trạng thái optimizer, scheduler, epoch để tiếp tục resume.
3. **`logs/`:** Chứa file `history.json` ghi lại tiến trình `train_loss`, `train_acc`, `val_f1` qua từng epoch.
4. **`calibration_images/`:** 500 ảnh mẫu được trích xuất tự động để phục vụ bước Lượng tử hóa PTQ trên NovaIC Toolchain của chip Novatek.
