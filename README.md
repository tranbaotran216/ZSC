# 🚀 Setup Environment for T2ICount + GroundingDINO

Hướng dẫn cài đặt môi trường và dependency cho project.

## 📁 1. Clone required repositories

```bash
cd zsc/T2ICount

git clone https://github.com/openai/CLIP.git clip
git clone https://github.com/CompVis/taming-transformers.git taming-transformers
```
⚠️ Lưu ý quan trọng
Hai thư mục clip và taming-transformers phải nằm trong: zsc/T2ICount/

Đổi tên thư mục: taming-transformers → taming (không đổi tên sẽ gây lỗi import)

## 📦 2. Tạo Conda Environment
Quay lại thư mục root (nơi chứa environment.yml):

Trong file environment.yml, dòng đầu tiên của pip:

```yaml
- --extra-index-url https://download.pytorch.org/whl/cu121
# dùng CUDA khác → sửa cu121 
```

🔧 Tạo environment
```bash
cd ../../
conda env create -n zsc -f environment.yml
conda activate zsc

cd CountGD/models/GroundingDINO/ops

#Thiết lập biến môi trường CUDA (Windows)
# Chỉnh lại path nếu CUDA của bạn nằm ở vị trí khác
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
$env:PATH += ";$env:CUDA_HOME\bin"

python setup.py build install
python test.py
# should result in 6 lines of * True
```


## 3. Run file
```
Run file train_fusion_arch_ver_3.sh
```
