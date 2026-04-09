

## 1. Tạo Conda Environment

Quay lại thư mục root (nơi chứa `environment.yml`)

Trong file `environment.yml` sẽ có dòng:

```yaml
- --extra-index-url https://download.pytorch.org/whl/cu121
# ⚠️ Nếu dùng CUDA khác → sửa cu121
```

---

### 🔧 Tạo environment

```bash
cd ../../

conda env create -n zsc -f environment.yml
conda activate zsc
cd CountGD\models\GroundingDINO\ops

# Thiết lập biến môi trường để tìm nvcc (thay đổi đường dẫn nếu cài ở vị trí khác) 
# choose the right cuda
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
$env:PATH += ";$env:CUDA_HOME\bin"

python setup.py build install
python test.py
# should result in 6 lines of * True
```


---

## ▶️ 3. Run file

```bash
run file train_fusion_arch_ver_3.sh

# trong file train_fusion_arch_ver_3.sh:
# - chỉnh lại PROJECT_DIR
# - FUSION_OPTION ở mục HYPERPARAMETERS: 'attn' sẽ đưa đưa features của unet và swin vào khối bidirectional cross attention + fusion
#                                        'base' sẽ fusion 2 features trước khi đưa vào feature enhancer
```

---

## ⚠️ Lưu ý

* Đảm bảo đã cài:

  * CUDA đúng version
  * Conda
* Nếu lỗi build:

  * Kiểm tra `nvcc --version`
  * Kiểm tra PATH CUDA


