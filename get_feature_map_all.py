import os
from nets.unet import Unet
import numpy as np
import torch
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names
from PIL import Image
import torchvision.transforms as transforms
def batch_process_images(input_folder, output_folder, model, transform):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历原图文件夹中的每张图片
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # 读取原图
            original_img = Image.open(os.path.join(input_folder, filename))

            # 图像转换和处理
            img = transform(original_img).unsqueeze(0)

            # 创建特征提取器
            feature_extractor = create_feature_extractor(model, return_nodes={"resnet.bn1": "output"})
            out = feature_extractor(img)

            # 提取并合并通道
            output = out["output"][0].detach().numpy()
            combined_output = np.sum(output, axis=0)

            # 缩放数据以适应8位范围[0, 255]
            scaled_output = ((combined_output - combined_output.min()) / (combined_output.max() - combined_output.min())) * 255
            scaled_output = scaled_output.astype(np.uint8)

            # 转换为PIL图像
            pil_image = Image.fromarray(scaled_output, mode='L')

            # 保存图像
            output_path = os.path.join(output_folder, filename)
            pil_image.save(output_path)

input_folder_path = r'E:\deeplearning\data\feature map\images'  # 原图所在文件夹
output_folder_path = r'E:\deeplearning\data\feature map\feature_map_all'  # 保存结果的文件夹

# 加载模型和转换器
model = Unet()
model.load_state_dict(torch.load(r'E:\deeplearning\program\UNet\model_data\unet_resnet_voc.pth'))
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 批量处理图片
batch_process_images(input_folder_path, output_folder_path, model, transform)