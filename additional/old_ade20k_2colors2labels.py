"""
STEP 2 in a database preprocessing:
	converting .png annotations (color maps) to numpy arrays (label maps) and saving them as .npy files
	(!!! this step can take several hours)

Color coding for ADE20K:
https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit#gid=0
(I exported this sheet as anno_filelist.csv file and used it in this script)
"""

import os
import cv2
import time
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from data.utils import colour2label

# -------------------------------------------------------------------------------------------------------------------- #
# Parameters and input data
# -------------------------------------------------------------------------------------------------------------------- #

inp_anno_csv = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/4_ADE20K/ADEChallengeData2016/anno_meta/anno_filelist.csv"
color_file = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/4_ADE20K/ADEChallengeData2016/color_coding.csv"
root_dir = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/4_ADE20K/ADEChallengeData2016"
# columns = ['idx', 'set', 'subdirs', 'filename_img', 'filename_seg', 'train']

# you can set to None next to variables if you don't want to change annotation file or make file with classes
new_column = "filename_lbl"
labels_file = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/4_ADE20K/ADEChallengeData2016/anno_meta/classes_map.csv"

# -------------------------------------------------------------------------------------------------------------------- #
# Prepare color dictionary and table
# -------------------------------------------------------------------------------------------------------------------- #

# First, get color-label mapping
df_colors = pd.read_csv(color_file)
rgb = df_colors["Color_Code (R,G,B)"]
rgb = rgb.str.replace("(", "").str.replace(")", "").str.split(",", expand=True).astype(int)
rgb = rgb.rename(columns={0: "r", 1: "g", 2: "b"})
df_colors = df_colors[["Idx", "Stuff", "Ratio", "Name"]]
df_colors = df_colors.join(rgb)
label_dict = {}
for _, row in df_colors.iterrows():
	r = row["r"]
	g = row["g"]
	b = row["b"]
	label_dict[(r, g, b)] = row["Idx"]

if labels_file:
	df_colors.to_csv(labels_file, index=False)

# -------------------------------------------------------------------------------------------------------------------- #
# Read data
# -------------------------------------------------------------------------------------------------------------------- #

# Then loop over segmentation maps
data = pd.read_csv(inp_anno_csv, index_col=0)
L = len(data)
L_digits = len(str(L))

if new_column:
	data[new_column] = None

# -------------------------------------------------------------------------------------------------------------------- #
# Loop over samples
# -------------------------------------------------------------------------------------------------------------------- #

sum_time = 0
for idx, row in data.iterrows():
	print("{}/{} ... ".format(str(idx+1).zfill(L_digits), L), end='')
	t = timer()

	path_seg = os.path.join(root_dir, row["set"], row["subdirs"], row["filename_seg"])
	seg = cv2.imread(path_seg)

	# DEBUG
	us = np.unique(seg, axis=0)

	# convert segmentation map from colours to labels and save it
	label_map = colour2label(image=seg, label_dict=label_dict, reverse_channels=True)

	out_npy_path = os.path.splitext(path_seg)[0] + '.npy'
	np.save(out_npy_path, label_map)

	t = int(round((timer() - t)*1000))
	sum_time += t
	remained = ((L-idx-1) / (idx+1) * sum_time) / 1000
	remained = time.strftime("%H:%M:%S", time.gmtime(remained))
	print("done! ({}x{}, \t{} ms, {}\tremains ~ {})".format(seg.shape[1],
															seg.shape[0],
															t,
															"\t" if t < 100 else "",
															remained))

	if new_column:
		data.loc[idx, new_column] = os.path.relpath(out_npy_path, root_dir)

# save modified annotation file (new column with path to label map was added)
if new_column:
	data.to_csv(inp_anno_csv)

