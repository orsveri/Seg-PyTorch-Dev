"""
==================================
||	Svetlana Orlova, 2020		||
||	github: orsveri				||
==================================

This is the implementation of UNet models.
Based on:
	[github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge],
which was based on this PyTorch implementation:
	[https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208]
TODO: now only model for input shape of 1024 implemented; add other options
"""

import torch
from torch import nn


class BlockDownsampling(nn.Module):

	def __init__(self, input_channels, output_channels, end_with_pooling=True):
		super(BlockDownsampling, self).__init__()
		self.end_with_pooling = end_with_pooling
		self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(3, 3), stride=1,
							   padding=1, bias=True, padding_mode='zeros')
		self.bn1 = nn.BatchNorm2d(num_features=output_channels)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=(3, 3), stride=1,
							   padding=1, bias=True, padding_mode='zeros')
		self.bn2 = nn.BatchNorm2d(num_features=output_channels)
		if end_with_pooling:
			self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x_link = self.relu(x)

		if self.end_with_pooling:
			x = self.pool(x_link)
			return x, x_link
		else:
			return x_link


class BlockUpsampling(nn.Module):

	def __init__(self, input_channels, output_channels):
		super(BlockUpsampling, self).__init__()
		self.up1 = nn.Upsample(size=(2, 2))
		self.conv1 = nn.Conv2d(in_channels=input_channels*2, out_channels=output_channels, kernel_size=(3, 3), stride=1)
		self.bn1 = nn.BatchNorm2d(num_features=output_channels)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(in_channels=input_channels * 2, out_channels=output_channels, kernel_size=(3, 3),
							   stride=1)
		self.bn2 = nn.BatchNorm2d(num_features=output_channels)
		self.conv3 = nn.Conv2d(in_channels=input_channels * 2, out_channels=output_channels, kernel_size=(3, 3),
							   stride=1)
		self.bn3 = nn.BatchNorm2d(num_features=output_channels)

	def forward(self, x, x_link):
		x = self.up1(x)
		x = torch.cat((x_link, x), dim=1) # concatenate along channels dimension
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		return x


class UNet(nn.Module):
	some_constants = None # TODO: remove if isn't needed

	def __init__(self, nb_classes, input_shape):
		super(UNet, self).__init__()
		self.down1 = BlockDownsampling(input_channels=3, output_channels=8, end_with_pooling=True)
		self.down2 = BlockDownsampling(input_channels=8, output_channels=16, end_with_pooling=True)
		self.down3 = BlockDownsampling(input_channels=16, output_channels=32, end_with_pooling=True)
		self.down4 = BlockDownsampling(input_channels=32, output_channels=64, end_with_pooling=True)
		self.down5 = BlockDownsampling(input_channels=64, output_channels=128, end_with_pooling=True)
		self.down6 = BlockDownsampling(input_channels=126, output_channels=256, end_with_pooling=True)
		self.down7 = BlockDownsampling(input_channels=256, output_channels=512, end_with_pooling=True)
		self.center = BlockDownsampling(input_channels=512, output_channels=1024, end_with_pooling=False)
		self.up7 = BlockUpsampling(input_channels=1024, output_channels=512)
		self.up6 = BlockUpsampling(input_channels=512, output_channels=256)
		self.up5 = BlockUpsampling(input_channels=256, output_channels=128)
		self.up4 = BlockUpsampling(input_channels=128, output_channels=64)
		self.up3 = BlockUpsampling(input_channels=64, output_channels=32)
		self.up2 = BlockUpsampling(input_channels=32, output_channels=16)
		self.up1 = BlockUpsampling(input_channels=16, output_channels=8)
		self.final_conv = nn.Conv2d(in_channels=8, out_channels=nb_classes, kernel_size=(1, 1), stride=1)

		if nb_classes == 1:
			self.final_activation = nn.Sigmoid()
		else:
			self.final_activation = nn.Softmax()

	def forward(self, x):
		x, x1_link = self.down1(x)
		x, x2_link = self.down2(x)
		x, x3_link = self.down3(x)
		x, x4_link = self.down4(x)
		x, x5_link = self.down5(x)
		x, x6_link = self.down6(x)
		x, x7_link = self.down7(x)
		x = self.center(x)
		x = self.up7(x=x, x_link=x7_link)
		x = self.up6(x=x, x_link=x6_link)
		x = self.up5(x=x, x_link=x5_link)
		x = self.up4(x=x, x_link=x4_link)
		x = self.up3(x=x, x_link=x3_link)
		x = self.up2(x=x, x_link=x2_link)
		x = self.up1(x=x, x_link=x1_link)
		x = self.final_conv(x)
		x = self.final_activation(x)
		return x


