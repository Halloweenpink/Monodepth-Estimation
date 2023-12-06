import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import os
from monodepth2.networks.depth_decoder import DepthDecoder
from monodepth2.networks.resnet_encoder import ResnetEncoder
from monodepth2.layers import *
def crop_center(img, crop_width, crop_height):
    img_height, img_width, _ = img.shape
    startx = img_width // 2 - (crop_width // 2)
    starty = img_height // 2 - (crop_height // 2)
    return img[starty:starty+crop_height, startx:startx+crop_width]

# 加载模型
encoder_path = "monodepth2/models/mono_640x192/encoder.pth"
depth_decoder_path = "monodepth2/models/mono_640x192/depth.pth"

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载编码器
encoder = ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

# 加载解码器
depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)
depth_decoder.to(device)
depth_decoder.eval()

folder = 'result_mono2_pic'
# folder = 'street_pic'
# 图像预处理
for i in range(1):
    num_pic = i
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    img = cv2.imread(f"../enhancedpics/{i+1}.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # input_image_resized = input_image.resize((feed_width, feed_height), Image.BILINEAR)
    input_image_resized = crop_center(img, feed_width, feed_height)

    # input_image_resized.show()
    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
    input_image_pytorch = input_image_pytorch.to(device)

    # 预测深度
    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp,
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        )

        # 将深度图转换为numpy数组
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        depth = 1 / disp_resized_np

    # 使用 Matplotlib 显示深度图
    plt.imshow(depth, cmap='plasma')
    plt.colorbar()
    plt.show()

    # 将深度图转换为灰度图以便保存
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype('uint8')
    depth_image = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)

    # 保存深度图
    filr_path = os.path.join(folder,f'depth_enhimage{num_pic+1}.jpg' )
    cv2.imwrite(filr_path, depth_image)
