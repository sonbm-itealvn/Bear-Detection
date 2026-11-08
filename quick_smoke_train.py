"""
quick_smoke_train.py - Script chạy thử nghiệm nhanh 1 epoch huấn luyện (Smoke / Dry-Run Test)
=============================================================================================
Mục đích:
  - Cho phép người dùng chạy thử nghiệm quy trình huấn luyện (Training Pipeline) ngay lập tức
    để kiểm tra xem mã nguồn, mô hình, tiền xử lý và hàm mất mát có hoạt động trơn tru không.
  - Tự động tạo một tập dữ liệu giả lập siêu nhỏ (Dummy dataset) gồm các ảnh RGB 640x360.
  - Chạy 1 epoch huấn luyện trên CPU hoặc GPU mà KHÔNG cần tải tập dữ liệu 100GB
    và KHÔNG cần tài khoản AWS Bedrock.
"""

import os
import sys
import shutil
import numpy as np
from pathlib import Path

def print_banner(text):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)

def main():
    print_banner("KIỂM TRA MÔI TRƯỜNG HUẤN LUYỆN (SMOKE TEST)")

    # 1. Kiểm tra Python & thư viện cơ bản
    try:
        import torch
        import torchvision
        from PIL import Image
        print(f"[OK] Python: {sys.version.split()[0]}")
        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"[OK] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[OK] Phát hiện {torch.cuda.device_count()} GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] Không có GPU NVIDIA -> Sẽ chạy thử nghiệm trên CPU.")
    except ImportError as e:
        print(f"[THIẾU THƯ VIỆN] {e}")
        print("\nVui lòng cài đặt môi trường trước khi chạy thử nghiệm:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  (hoặc bản cpu: pip install torch torchvision)")
        print("  pip install timm pillow numpy scipy opencv-python tqdm\n")
        sys.exit(1)

    # 2. Tạo tập dữ liệu Dummy mini (10 ảnh train, 4 ảnh val)
    dummy_dir = Path("dummy_smoke_dataset")
    print_banner("1. TẠO TẬP DỮ LIỆU THỬ NGHIỆM TỰ ĐỘNG (DUMMY DATASET)")
    
    splits = {
        "train": {"bear": 6, "non_bear": 6},
        "val": {"bear": 2, "non_bear": 2}
    }
    
    for split, classes in splits.items():
        for cls, count in classes.items():
            target_path = dummy_dir / split / cls
            target_path.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img_path = target_path / f"{cls}_{i:03d}.jpg"
                if not img_path.exists():
                    # Tạo ảnh ngẫu nhiên kích thước 640x360
                    # Thêm chút màu đặc trưng (gấu nâu/đen, non_bear sáng hơn)
                    if cls == "bear":
                        arr = np.random.randint(20, 80, (360, 640, 3), dtype=np.uint8)
                    else:
                        arr = np.random.randint(100, 220, (360, 640, 3), dtype=np.uint8)
                    Image.fromarray(arr).save(img_path)
    
    print(f"[OK] Đã khởi tạo dữ liệu mẫu tại: {dummy_dir.resolve()}")
    print("     - Train: 6 ảnh bear, 6 ảnh non_bear")
    print("     - Val:   2 ảnh bear, 2 ảnh non_bear")

    # 3. Cấu hình biến môi trường cho chạy thử nghiệm
    run_dir = Path("quick_smoke_run")
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    os.environ["OVERRIDE_EPOCHS"] = "1"              # Chỉ chạy 1 epoch thử nghiệm
    os.environ["OVERRIDE_BATCH_SIZE"] = "2"          # Batch size nhỏ cho máy CPU/ít RAM
    os.environ["OVERRIDE_NUM_WORKERS"] = "0"         # 0 worker tránh lỗi multiprocessing trên Windows
    os.environ["OVERRIDE_HEM_START"] = "999"         # Tắt HEM inference pass để chạy siêu tốc
    os.environ["OVERRIDE_RUN_DIR"] = str(run_dir)    # Thư mục lưu kết quả thử nghiệm
    if not torch.cuda.is_available():
        os.environ["OVERRIDE_CUDA_DEVICES"] = ""

    print_banner("2. BẮT ĐẦU CHẠY HUẤN LUYỆN THỬ NGHIỆM (1 EPOCH)")
    
    cmd_args = [
        sys.executable,
        "train_mobilenetv2_relu_novaic_23_4.py",
        "--data-dir", str(dummy_dir),
        "--no-claude-verify",                         # Tắt xác thực AWS Bedrock
    ]
    
    print("Lệnh thực thi:")
    print("  " + " ".join(cmd_args))
    print("-" * 70)

    import subprocess
    ret = subprocess.run(cmd_args)

    if ret.returncode == 0:
        print_banner("3. KẾT QUẢ CHẠY THỬ NGHIỆM: THÀNH CÔNG RỰC RỠ!")
        print(f"[OK] Model checkpoint đã được tạo tại thư mục: {run_dir / 'checkpoints'}")
        print(f"[OK] Thư mục weights: {run_dir / 'models'}")
        print("\n=> Pipeline huấn luyện MobileNetV2-ReLU / RegNetX hoạt động hoàn hảo và sẵn sàng!")
    else:
        print_banner("3. KẾT THÚC VỚI MÃ LỖI:")
        print(f"Lỗi trả về (Exit Code): {ret.returncode}")

if __name__ == "__main__":
    main()
