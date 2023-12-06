import os
import cv2
import torch
from matplotlib import pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 选择模型类型
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

# 加载模型
model = torch.hub.load("isl-org/MiDaS", model_type)
model.to(device)
model.eval()

# 加载变换
midas_transforms = torch.hub.load("isl-org/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# 结果保存目录
folder = 'result_midas_pic'
if not os.path.exists(folder):
    os.makedirs(folder)

# 初始化摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头的一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换颜色空间
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 应用预处理并生成深度图
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # 转换深度图并保存
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype('uint8')
    depth_image = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)

    # 显示结果
    cv2.imshow('Depth Image', depth_image)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
