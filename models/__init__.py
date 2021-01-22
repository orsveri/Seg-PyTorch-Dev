from.utils import save_checkpoint

from .unet import UNet
from torchvision.models.segmentation import deeplabv3_resnet50

model_dict = {"unet": UNet,
			  "deeplabv3_resnet50": deeplabv3_resnet50}

def get_model_constructor(model_name):
	if model_name in model_dict:
		return model_dict[model_name]
	else:
		print("No model with name {}! Model constructor wasn't returned".format(model_name))