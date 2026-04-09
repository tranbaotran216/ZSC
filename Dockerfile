# Sử dụng base image có sẵn CUDA và cuDNN phù hợp với cu121
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Thiết lập biến môi trường để không bị dừng khi hỏi vùng miền (timezone)
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và build C++
RUN apt-get update && apt-get install -y \
    git \
    make \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file environment.yml (nếu bạn muốn dùng conda trong docker) 
# Hoặc cài trực tiếp qua pip để image nhẹ hơn. Ở đây ta dùng pip.
COPY . .

# 1. Cài đặt các dependency chính
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir ftfy regex tqdm

# 2. Setup repositories con (theo hướng dẫn của bạn)
RUN cd zsc/T2ICount && \
    git clone https://github.com/openai/CLIP.git clip && \
    git clone https://github.com/CompVis/taming-transformers.git taming

# 3. Build GroundingDINO CUDA Ops
# Thiết lập biến môi trường cho quá trình build
ENV CUDA_HOME=/usr/local/cuda
RUN cd CountGD/models/GroundingDINO/ops && \
    python setup.py build install

# Kiểm tra thử build thành công không (Optional)
RUN cd CountGD/models/GroundingDINO/ops && python test.py

# Lệnh mặc định khi chạy container
CMD ["python", "your_main_script.py"]