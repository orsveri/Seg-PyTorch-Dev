import cv2
import numpy as np

def to_numpy(tensor):
	return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def advanced_resize(img, target_h, target_w, keep_asp_ratio=False, nearest=False, relation_minmax = (0.99, 1.01),
					color=(0, 0, 0)):
	"""
	resize image with some options: keeping aspect ratio (padding with given color), using interpolation with nearest
	neighbour (for segmentation masks) or bicubic (for normal images)
	:param img: numpy.ndarray,
	:param target_h: int, target height
	:param target_w: int, target width
	:param keep_asp_ratio: bool, if True, padding will be used, if False - image will be resized without keeping the
			aspect ratio
	:param nearest: bool, if True, nearest neighbour interpolation will be used for resize, if False - bicubic
	:param relation_minmax: list or tuple with two values: min and max relation asp_ratio_target/asp_ratio_img
	:param color: color for padding area
	:return: resized image
	"""
	#assert len(img.shape) == 3, "tools.data.utils.advanced_resize: img.shape must have length 3!"

	method = cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC

	asp_ratio_img = img.shape[0] / img.shape[1]
	asp_ratio_target = target_h / target_w
	relation = asp_ratio_target / asp_ratio_img

	# if aspect ratio of image is not like target one and we don't want to deform the image
	if not (relation_minmax[0] <= relation <= relation_minmax[1]) and keep_asp_ratio:
		# padding for getting target aspect ratio
		if asp_ratio_img < asp_ratio_target:
			dh = round(img.shape[1] * asp_ratio_target - img.shape[0])
			img = cv2.copyMakeBorder(img, top=int(dh // 2), bottom=int(dh // 2 + dh % 2), left=0, right=0,
									 borderType=cv2.BORDER_CONSTANT, value=color)
		else:
			dw = round(img.shape[0] / asp_ratio_target - img.shape[1])
			img = cv2.copyMakeBorder(img, left=int(dw // 2), right=int(dw // 2 + dw % 2), top=0, bottom=0,
									 borderType=cv2.BORDER_CONSTANT, value=color)

	# finally, resize
	return cv2.resize(img, (target_w, target_h), method)


def colour2label(image, label_dict, reverse_channels=False):
	"""
	# TODO: optimize
	takes an image (segmentation map) and a dictionary of category colour values.
	Replaces colours with labels (label is the index of a colour in the list).
	:param image: 			 numpy-array (segmentation map) with shape (H, W, C), where C - channels.
	:param label_dict: 		 dictionary with colours as tuple keys and labels as values.
	:param reverse_channels: boolean flag, True if image channels are BGR, False if RGB.
	:return:				 label map with shape (H, W), each pixel is assigned a label.
	"""
	rows, cols, channels = image.shape
	if reverse_channels:
		image = image[:,:,::-1]
	label_map = np.zeros(shape=(rows, cols), dtype=np.int)

	for row in range(rows):
		for col in range(cols):
			pixel = tuple(image[row, col])
			if pixel in label_dict:
				label_map[row, col] = label_dict[pixel]

	return label_map



