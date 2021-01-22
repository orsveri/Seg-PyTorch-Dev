"""
Dataset classes for different datasets (databases).
Dataset class implements methods __len__ and __getitem__
"""
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data.utils import advanced_resize


class Dataset_ADE20K(Dataset):
	"""ADE20K dataset"""

	def __init__(self, csv_file, root_dir, nb_classes, input_shape=None, resize_pad=False, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Root directory for the dataset.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.data = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.nb_classes = nb_classes
		self.transform = transform
		self.resize_pad = resize_pad
		if isinstance(input_shape, (tuple, list)) and 2 <= len(input_shape) <= 3:
			self.input_shape = input_shape
		else:
			self.input_shape = None

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		data_item = self.data.iloc[idx]
		path_img = os.path.join(self.root_dir, data_item["subdirs_img"], data_item["filename_img"])
		path_lbl = os.path.join(self.root_dir, data_item["subdirs_seg"], data_item["filename_seg"])

		# read input image
		img = cv2.imread(path_img)
		# read label map, it's in .npy file
		lbl_ = cv2.imread(path_lbl, cv2.IMREAD_GRAYSCALE)
		# a little preparation for label map
		lbl_ = lbl_.astype(np.uint8)
		#if len(lbl.shape) == 2:
		#	lbl = np.expand_dims(lbl, axis=-1)

		# image preprocessing: resizing, later - transformations
		if self.input_shape:
			img = advanced_resize(img=img, target_h=self.input_shape[0], target_w=self.input_shape[1],
								  keep_asp_ratio=True, nearest=False)
			lbl_ = advanced_resize(img=lbl_, target_h=self.input_shape[0], target_w=self.input_shape[1],
								  keep_asp_ratio=True, nearest=True)

		# one-hot decoding label masks TODO: is it optimal?
		# https://gist.github.com/frnsys/91a69f9f552cbeee7b565b3149f29e3e
		lbl = np.zeros((lbl_.shape[0], lbl_.shape[1], self.nb_classes+1))
		class_idx = np.arange(lbl_.shape[0]).reshape(lbl_.shape[0], 1)
		component_idx = np.tile(np.arange(lbl_.shape[1]), (lbl_.shape[0], 1))
		lbl[class_idx, component_idx, lbl_] = 1
		# we don't need channel for index=0, it isn't category
		lbl = lbl[:,:,1:]

		# To Pytorch tensor (C, H, W)
		img = torch.from_numpy(np.transpose(img, axes=(2, 0, 1))).type(torch.FloatTensor)
		lbl = torch.from_numpy(np.transpose(lbl, axes=(2, 0, 1))).type(torch.FloatTensor)

		sample = [img, lbl]

		if self.transform:
			sample = self.transform(sample)

		return sample


