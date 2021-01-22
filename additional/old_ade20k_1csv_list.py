"""
STEP 1 in a database preprocessing:
	making a list of all input image and annotation files and saving it as .csv file
"""

import os
import scipy.io
import numpy as np
import pandas as pd


rootdir = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/4_ADE20K/ADE20K_2016_07_26/"
out_anno_subdir = "anno_meta"
out_anno_csv = "anno_filelist.csv"
img_sets = {"images/training": "train", "images/validation": "val"}

seg_suffix = "_seg.png"
columns = ['set', 'subdirs', 'filename_img', 'filename_seg', 'train']
data_list = []

L_iss = len(img_sets)
for i_iss, img_set_subdir in enumerate(img_sets):
	set_subdir = os.path.join(rootdir, img_set_subdir)
	file_groups = list(os.walk(set_subdir))
	L_fg = len(file_groups)
	for i_fg, file_group in enumerate(file_groups):
		print("{}/{} \t {}/{} \t processing.. ".format(i_iss+1, L_iss, i_fg+1, L_fg), end='')
		if len(file_group[1]) > 0:
			print('skipped')
			continue
		subdir = os.path.relpath(file_group[0], set_subdir)
		for f in file_group[2]:
			img_name, img_ext = os.path.splitext(f)
			if img_ext != '.jpg':
				continue
			f_seg = img_name + seg_suffix
			if f_seg in file_group[2]:
				data_list.append([img_set_subdir, subdir, f, f_seg, img_sets[img_set_subdir]=="train"])
		print('done')

data_list = pd.DataFrame(data_list, columns=columns)
os.makedirs(os.path.join(rootdir, out_anno_subdir), exist_ok=True)
data_list.to_csv(os.path.join(rootdir, out_anno_subdir, out_anno_csv))

print('')


