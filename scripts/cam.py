from torchvision.models import resnet18,resnet50
import segmentation_models_pytorch as smp
import torchvision.models.segmentation as seg
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
import matplotlib.pyplot as plt
import torch

def hook(module, input, output):
    # 在这里对输入或输出进行处理，这里以输出为例
    print("Layer输出:", output.shape)

# 定义模型
model = resnet50(pretrained=True).eval()
# chpt_path = ''
# sd = torch.load(chpt_path, map_location='cpu')['state_dict']
# keys = list(sd.keys())
# model.load_state_dict(sd, strict=False)

cam_extractor = GradCAMpp(model,target_layer='layer4')
# cam_extractor = SmoothGradCAMpp(model,target_layer='decoder')

# 读取图片
img = read_image("../data/cat/T0001.jpg")
input_tensor = normalize(resize(img, (128, 128)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# 获得模型的输出
out = model(input_tensor.unsqueeze(0))

# 将模型的输出输入到cam_extractor
activation_map = cam_extractor(out.squeeze(0).argmax().item(),out)

# 可视化激活图
plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()

# 可视化叠加图：
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()