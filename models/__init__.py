from .unet import UNet

model_dict = {"unet": UNet}

def get_model_constructor(model_name):
	if model_name in model_dict:
		return model_dict[model_name]
	else:
		print("No model with name {}! Model constructor wasn't returned".format(model_name))