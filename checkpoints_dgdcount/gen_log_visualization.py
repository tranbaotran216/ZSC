import re
import pandas as pd
import matplotlib.pyplot as plt

# 1. Đọc nội dung tệp log
file_path = 'D:/projects/zsc/checkpoints_dgdcount/model_arch_ver_2/unet_gdino_text_only_arch_ver_2.log'
with open(file_path, 'r', encoding='utf-8') as f:
    log_content = f.read()


pattern = r"--- Epoch (\d+) Results ---\nTrain Loss: ([\d.]+) \| MAE: ([\d.]+)\nVal Loss: ([\d.]+) \| MAE: ([\d.]+)\nVal MAE: ([\d.]+) \| RMSE: ([\d.]+)"
matches = re.findall(pattern, log_content)

data = []
for m in matches:
    data.append({
        'epoch': int(m[0]),
        'train_loss': float(m[1]),
        'train_mae': float(m[2]),
        'val_loss': float(m[3]),
        'val_mae': float(m[5]), # Lấy giá trị Val MAE ở dòng riêng biệt
        'rmse': float(m[6])
    })

# Tạo DataFrame
df = pd.DataFrame(data)

# Lưu dữ liệu ra CSV để theo dõi nếu cần
df.to_csv('D:/projects/zsc/checkpoints_dgdcount/model_arch_ver_2/training_metrics_arch_ver_2.csv', index=False)
print("Đã trích xuất dữ liệu thành công (28 epochs).")

# 3. Vẽ biểu đồ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Biểu đồ Loss
ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', color='#1f77b4')
ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o', color='#ff7f0e')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Biểu đồ MAE
ax2.plot(df['epoch'], df['train_mae'], label='Train MAE', marker='s', color='#1f77b4')
ax2.plot(df['epoch'], df['val_mae'], label='Val MAE', marker='s', color='#ff7f0e')
ax2.set_title('Training and Validation MAE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('training_plots.png')
plt.show()