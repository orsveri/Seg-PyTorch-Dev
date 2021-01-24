from .datasets import Dataset_ADE20K


dataset_dict = {"ade20k": Dataset_ADE20K}

# TODO: finish
def get_dataset_constructor(dataset_name):
	if dataset_name in dataset_dict:
		return dataset_dict[dataset_name]
	else:
		print("No dataset with name {}! Dataset constructor has not been returned".format(dataset_name))