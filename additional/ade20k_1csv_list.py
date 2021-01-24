"""
ADE20K Dataset, http://sceneparsing.csail.mit.edu/ (scene Parsing train/val data)

STEP 1 in a database preprocessing:
	making a list of all input image and annotation files and saving it as .csv file
"""

import os
import pandas as pd


rootdir = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/4_ADE20K/ADEChallengeData2016/"
out_anno_subdir = "anno_meta"
out_anno_csv = "anno_filelist.csv"
# required format: {"train": {"images": "<..>", "masks": "<..>"}, "val": {"images": "<..>", "masks": "<..>"}}
data_sets = {"train": {"images": "images/training", "masks": "annotations/training"},
			"val": {"images": "images/validation", "masks": "annotations/validation"}}
columns = ["set", "subdirs_img", "subdirs_seg", "filename_img", "filename_seg", "train"]
img_ext = ".jpg"
seg_ext = ".png"
data_list = []

L_iss = len(data_sets)
for i_iss, key in enumerate(data_sets):
	data_set = data_sets[key]
	subdir_img = os.path.join(rootdir, data_set["images"])
	subdir_seg = os.path.join(rootdir, data_set["masks"])
	img_list = [i for i in os.listdir(subdir_img) if i.endswith(img_ext)]
	L_img = len(img_list)
	for i_im, img_name in enumerate(img_list):
		print("{}/{} \t {}/{} \t processing.. ".format(i_iss+1, L_iss, i_im+1, L_img), end='')
		# check if there is corresponding annotation file
		seg_name = os.path.splitext(img_name)[0] + seg_ext
		if not os.path.isfile(os.path.join(subdir_seg, seg_name)):
			print("segmentation mask haven't been found! Skip")
			continue

		data_list.append([key, data_set["images"], data_set["masks"], img_name, seg_name, key == "train"])
		print('done')

data_list = pd.DataFrame(data_list, columns=columns)
os.makedirs(os.path.join(rootdir, out_anno_subdir), exist_ok=True)
data_list.to_csv(os.path.join(rootdir, out_anno_subdir, out_anno_csv))

print('Everything is done!')


