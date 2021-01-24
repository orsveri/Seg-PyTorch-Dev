from.utils import save_checkpoint

from .unet import UNet
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

model_dict = {"unet": {"constructor": UNet, "weights": None},
			  "deeplabv3_resnet50": {"constructor": deeplabv3_resnet50, "weights": None},
			  "deeplabv3_resnet101": {"constructor": deeplabv3_resnet101, "weights": "deeplabv3_resnet101_coco-586e9e4e.pth"}}

def get_model_constructor(model_name):
	if model_name in model_dict:
		return model_dict[model_name]
	else:
		print("No model with name {}! Model constructor has not been returned".format(model_name))