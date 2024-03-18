import torchvision
import torch
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from matplotlib import pyplot as plt
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary
import os
import numpy as np
from torchvision.utils import save_image
from nets.unet import Unet



model = Unet()
batch_size = 1
summary(model, input_size=(batch_size, 3, 512, 512))
print(summary)
# nodes, _ = get_graph_node_names(model)


transform = transforms.Compose([
                                transforms.CenterCrop(512),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
model = Unet()
model.load_state_dict(torch.load(r'E:\deeplearning\program\UNet\model_data\unet_resnet_voc.pth'))

original_img = Image.open(r"E:\deeplearning\data\feature map\hand_make_crop\images-200\CNV-163081-23.png")
image = original_img.resize((512,512), Image.BICUBIC)
img = transform(image).unsqueeze(0)

figure_save_path = "E:/deeplearning/data/feature map/CNV-135126-3"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建

# nodes, _ = get_graph_node_names(model)

# ------------单层，合并通道输出--------------
feature_extractor = create_feature_extractor(model, return_nodes={"inputs": "output"})
out = feature_extractor(img)

output = out["output"][0].detach().numpy()  # Assuming batch size is 1
combined_output = np.sum(output, axis=0)  # Combine all channels

# Scale the data to fit within 8-bit range [0, 255]
scaled_output = ((combined_output - combined_output.min()) / (combined_output.max() - combined_output.min())) * 255
scaled_output = scaled_output.astype(np.uint8)  # Convert to 8-bit integer

# Convert NumPy array to PIL Image
pil_image = Image.fromarray(scaled_output, mode='L')  # 'L' mode for grayscale image

last_path = os.path.join(figure_save_path, 'unet_combined')
if not os.path.exists(last_path):
    os.makedirs(last_path)

# Save the image without changing its size
pil_image.save(os.path.join(last_path, 'resnet.inputs_combined_output.png'))
'''
# ------------所有层，合并通道输出--------------
for node in nodes:
    feature_extractor = create_feature_extractor(model, return_nodes={node: "output"})
    out = feature_extractor(img)

    output = out["output"][0].detach().numpy()  # Assuming batch size is 1
    combined_output = np.sum(output, axis=0)  # Combine all channels
    
    # Scale the data to fit within 8-bit range [0, 255]
    scaled_output = ((combined_output - combined_output.min()) / (combined_output.max() - combined_output.min())) * 255
    scaled_output = scaled_output.astype(np.uint8)  # Convert to 8-bit integer

    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(scaled_output, mode='L')  # 'L' mode for grayscale image
   
    last_path = os.path.join(figure_save_path, 'unet_combined')
    if not os.path.exists(last_path):
        os.makedirs(last_path)
        
    # Save the image without changing its size
    pil_image.save(os.path.join(last_path, f'{node}_combined_output.png'))
'''
'''
# ------------分通道输出--------------
for node in nodes:
    feature_extractor = create_feature_extractor(model, return_nodes={node: "output"})
    out = feature_extractor(img)

    output = out["output"][0]  # Assuming batch size is 1
    num_channels = output.shape[0]

    last_path = os.path.join(figure_save_path, 'unet_all_channels', f'{node}')
    if not os.path.exists(last_path):
        os.makedirs(last_path)

    for i in range(num_channels):
        img_out=output[i].detach().numpy()
        scaled_output = ((img_out - img_out.min()) / (img_out.max() - img_out.min())) * 255
        scaled_output = scaled_output.astype(np.uint8)  # Convert to 8-bit integer
        pil_image = Image.fromarray(scaled_output, mode='L')  # 'L' mode for grayscale image

        # Save the image without changing its size
        pil_image.save(os.path.join(last_path, f'{node}_output_{i}.png'))

'''

'''
# return_nodes参数就是返回对应的输出
feature_extractor = create_feature_extractor(model, return_nodes={"resnet.layer1.0.conv2":"output"})

out = feature_extractor(img)

# 这里没有分通道可视化
plt.imshow(out["output"][0].transpose(0, 1).sum(1).detach().numpy())
plt.axis('off')  # Turn off axis labels and ticks
plt.savefig(os.path.join(figure_save_path, 'resnet.layer1.0.conv2.png'), bbox_inches='tight', pad_inches=0)
plt.close()  # Close the plot to prevent displaying it
'''

'''
output_img = Image.fromarray(np.uint8(out["output"][0].transpose(0, 1).sum(1).detach().numpy()))
output_img.show()
'''
#output_img.save(os.path.join(figure_save_path, 'resnet.conv1.png'))
#print(nodes)
"""
-------------输出----------------
 ['inputs'
 'resnet.conv1'
 'resnet.bn1'
 'resnet.relu'
 'resnet.maxpool'
 'resnet.layer1.0.conv1'
 'resnet.layer1.0.bn1'
 'resnet.layer1.0.relu'
 'resnet.layer1.0.conv2'
 'resnet.layer1.0.bn2'
 'resnet.layer1.0.relu_1'
 'resnet.layer1.0.conv3'
 'resnet.layer1.0.bn3'
 'resnet.layer1.0.downsample.0'
 'resnet.layer1.0.downsample.1'
 'resnet.layer1.0.add'
 'resnet.layer1.0.relu_2'
 'resnet.layer1.1.conv1'
 'resnet.layer1.1.bn1'
 'resnet.layer1.1.relu'
 'resnet.layer1.1.conv2'
 'resnet.layer1.1.bn2'
 'resnet.layer1.1.relu_1'
 'resnet.layer1.1.conv3'
 'resnet.layer1.1.bn3'
 'resnet.layer1.1.add'
 'resnet.layer1.1.relu_2'
 'resnet.layer1.2.conv1'
 'resnet.layer1.2.bn1'
 'resnet.layer1.2.relu'
 'resnet.layer1.2.conv2'
 'resnet.layer1.2.bn2'
 'resnet.layer1.2.relu_1'
 'resnet.layer1.2.conv3'
 'resnet.layer1.2.bn3'
 'resnet.layer1.2.add'
 'resnet.layer1.2.relu_2'
 'resnet.layer2.0.conv1'
 'resnet.layer2.0.bn1'
 'resnet.layer2.0.relu'
 'resnet.layer2.0.conv2'
 'resnet.layer2.0.bn2'
 'resnet.layer2.0.relu_1'
 'resnet.layer2.0.conv3'
 'resnet.layer2.0.bn3'
 'resnet.layer2.0.downsample.0'
 'resnet.layer2.0.downsample.1'
 'resnet.layer2.0.add'
 'resnet.layer2.0.relu_2'
 'resnet.layer2.1.conv1'
 'resnet.layer2.1.bn1'
 'resnet.layer2.1.relu'
 'resnet.layer2.1.conv2'
 'resnet.layer2.1.bn2'
 'resnet.layer2.1.relu_1'
 'resnet.layer2.1.conv3'
 'resnet.layer2.1.bn3'
 'resnet.layer2.1.add'
 'resnet.layer2.1.relu_2'
 'resnet.layer2.2.conv1'
 'resnet.layer2.2.bn1'
 'resnet.layer2.2.relu'
 'resnet.layer2.2.conv2'
 'resnet.layer2.2.bn2'
 'resnet.layer2.2.relu_1'
 'resnet.layer2.2.conv3'
 'resnet.layer2.2.bn3'
 'resnet.layer2.2.add'
 'resnet.layer2.2.relu_2'
 'resnet.layer2.3.conv1'
 'resnet.layer2.3.bn1'
 'resnet.layer2.3.relu'
 'resnet.layer2.3.conv2'
 'resnet.layer2.3.bn2'
 'resnet.layer2.3.relu_1'
 'resnet.layer2.3.conv3'
 'resnet.layer2.3.bn3'
 'resnet.layer2.3.add'
 'resnet.layer2.3.relu_2'
 'resnet.layer3.0.conv1'
 'resnet.layer3.0.bn1'
 'resnet.layer3.0.relu'
 'resnet.layer3.0.conv2'
 'resnet.layer3.0.bn2'
 'resnet.layer3.0.relu_1'
 'resnet.layer3.0.conv3'
 'resnet.layer3.0.bn3'
 'resnet.layer3.0.downsample.0'
 'resnet.layer3.0.downsample.1'
 'resnet.layer3.0.add'
 'resnet.layer3.0.relu_2'
 'resnet.layer3.1.conv1'
 'resnet.layer3.1.bn1'
 'resnet.layer3.1.relu'
 'resnet.layer3.1.conv2'
 'resnet.layer3.1.bn2'
 'resnet.layer3.1.relu_1'
 'resnet.layer3.1.conv3'
 'resnet.layer3.1.bn3'
 'resnet.layer3.1.add'
 'resnet.layer3.1.relu_2'
 'resnet.layer3.2.conv1'
 'resnet.layer3.2.bn1'
 'resnet.layer3.2.relu'
 'resnet.layer3.2.conv2'
 'resnet.layer3.2.bn2'
 'resnet.layer3.2.relu_1'
 'resnet.layer3.2.conv3'
 'resnet.layer3.2.bn3'
 'resnet.layer3.2.add'
 'resnet.layer3.2.relu_2'
 'resnet.layer3.3.conv1'
 'resnet.layer3.3.bn1'
 'resnet.layer3.3.relu'
 'resnet.layer3.3.conv2'
 'resnet.layer3.3.bn2'
 'resnet.layer3.3.relu_1'
 'resnet.layer3.3.conv3'
 'resnet.layer3.3.bn3'
 'resnet.layer3.3.add'
 'resnet.layer3.3.relu_2'
 'resnet.layer3.4.conv1'
 'resnet.layer3.4.bn1'
 'resnet.layer3.4.relu'
 'resnet.layer3.4.conv2'
 'resnet.layer3.4.bn2'
 'resnet.layer3.4.relu_1'
 'resnet.layer3.4.conv3'
 'resnet.layer3.4.bn3'
 'resnet.layer3.4.add'
 'resnet.layer3.4.relu_2'
 'resnet.layer3.5.conv1'
 'resnet.layer3.5.bn1'
 'resnet.layer3.5.relu'
 'resnet.layer3.5.conv2'
 'resnet.layer3.5.bn2'
 'resnet.layer3.5.relu_1'
 'resnet.layer3.5.conv3'
 'resnet.layer3.5.bn3'
 'resnet.layer3.5.add'
 'resnet.layer3.5.relu_2'
 'resnet.layer4.0.conv1'
 'resnet.layer4.0.bn1'
 'resnet.layer4.0.relu'
 'resnet.layer4.0.conv2'
 'resnet.layer4.0.bn2'
 'resnet.layer4.0.relu_1'
 'resnet.layer4.0.conv3'
 'resnet.layer4.0.bn3'
 'resnet.layer4.0.downsample.0'
 'resnet.layer4.0.downsample.1'
 'resnet.layer4.0.add'
 'resnet.layer4.0.relu_2'
 'resnet.layer4.1.conv1'
 'resnet.layer4.1.bn1'
 'resnet.layer4.1.relu'
 'resnet.layer4.1.conv2'
 'resnet.layer4.1.bn2'
 'resnet.layer4.1.relu_1'
 'resnet.layer4.1.conv3'
 'resnet.layer4.1.bn3'
 'resnet.layer4.1.add'
 'resnet.layer4.1.relu_2'
 'resnet.layer4.2.conv1'
 'resnet.layer4.2.bn1'
 'resnet.layer4.2.relu'
 'resnet.layer4.2.conv2'
 'resnet.layer4.2.bn2'
 'resnet.layer4.2.relu_1'
 'resnet.layer4.2.conv3'
 'resnet.layer4.2.bn3'
 'resnet.layer4.2.add'
 'resnet.layer4.2.relu_2'
 'up_concat4.up'
 'up_concat4.cat'
 'up_concat4.conv1'
 'up_concat4.relu'
 'up_concat4.conv2'
 'up_concat4.relu_1'
 'up_concat3.up'
 'up_concat3.cat'
 'up_concat3.conv1'
 'up_concat3.relu'
 'up_concat3.conv2'
 'up_concat3.relu_1'
 'up_concat2.up'
 'up_concat2.cat'
 'up_concat2.conv1'
 'up_concat2.relu'
 'up_concat2.conv2'
 'up_concat2.relu_1'
 'up_concat1.up'
 'up_concat1.cat'
 'up_concat1.conv1'
 'up_concat1.relu'
 'up_concat1.conv2'
 'up_concat1.relu_1'
 'up_conv.0'
 'up_conv.1'
 'up_conv.2'
 'up_conv.3'
 'up_conv.4'
 'final']
"""
